// Local, in-process inspector evaluation; no debugging port is opened.
// REPL mode preserves lexical bindings and permits top-level await.
(() => {
  const { Session } = require('node:inspector');
  const { createInterface } = require('node:readline');
  const { createRequire } = require('node:module');
  const { resolve } = require('node:path');
  const { pathToFileURL } = require('node:url');
  const { inspect } = require('node:util');
  const { AsyncLocalStorage } = require('node:async_hooks');
  const session = new Session();
  session.connect();
  const post = (method, params) => new Promise((resolve, reject) =>
    session.post(method, params, (error, result) => error ? reject(error) : resolve(result)));
  const write = process.stdout.write.bind(process.stdout);
  const emit = (value) => write(JSON.stringify(value) + '\n');
  const cells = new AsyncLocalStorage();
  const limit = 64 * 1024;
  for (const stream of [process.stdout, process.stderr]) {
    stream.write = (chunk, encoding, callback) => {
      const cell = cells.getStore();
      if (cell && !cell.done && cell.bytes < limit) {
        const text = Buffer.isBuffer(chunk) ? chunk.toString('utf8') : String(chunk);
        const clipped = text.slice(0, limit - cell.bytes);
        cell.bytes += Buffer.byteLength(clipped);
        emit({ id: cell.id, output: clipped });
        if (cell.bytes >= limit) emit({ id: cell.id, output: '\n[Node output truncated at 64 KiB]\n' });
      }
      const cb = typeof encoding === 'function' ? encoding : callback;
      if (cb) queueMicrotask(cb);
      return true;
    };
  }
  globalThis.require = createRequire(resolve(process.cwd(), '__optimus_node__.cjs'));
  // Dynamic import inside inspector expressions has no module loader callback.
  // This helper imports through the real Node module loader instead.
  globalThis.nodeImport = (specifier) => import(specifier.startsWith('.') || specifier.startsWith('/')
    ? pathToFileURL(resolve(process.cwd(), specifier)).href
    : specifier.startsWith('node:') ? specifier : pathToFileURL(globalThis.require.resolve(specifier)).href);
  globalThis.__optimusInspect = (value) => inspect(value, {
    colors: false, depth: 4, maxArrayLength: 100, maxStringLength: 8192, customInspect: false, getters: false,
  });
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  let queue = Promise.resolve();
  input.on('line', (line) => {
    queue = queue.then(async () => {
      const request = JSON.parse(line);
      const cell = { id: request.id, bytes: 0, done: false };
      await cells.run(cell, async () => {
        try {
          const response = await post('Runtime.evaluate', {
            expression: request.code, replMode: true, awaitPromise: true, objectGroup: 'optimus-cell',
          });
          let text = '';
          if (response.exceptionDetails) {
            text = response.exceptionDetails.exception?.description || response.exceptionDetails.text;
          } else if (response.result.type !== 'undefined') {
            if (response.result.objectId) {
              const formatted = await post('Runtime.callFunctionOn', {
                objectId: response.result.objectId,
                functionDeclaration: 'function() { return globalThis.__optimusInspect(this); }', returnByValue: true,
              });
              text = formatted.result.value || response.result.description || '';
            } else {
              text = response.result.unserializableValue ?? inspect(response.result.value, { colors: false, maxStringLength: 8192 });
            }
          }
          emit({ id: request.id, done: true, error: !!response.exceptionDetails, result: text.slice(0, limit) });
        } catch (error) {
          emit({ id: request.id, done: true, error: true, result: String(error).slice(0, limit) });
        } finally {
          cell.done = true;
          await post('Runtime.releaseObjectGroup', { objectGroup: 'optimus-cell' });
          await post('Runtime.discardConsoleEntries', {});
        }
      });
    }).catch((error) => { emit({ fatal: String(error) }); process.exitCode = 1; input.close(); });
  });
  input.on('close', () => process.exit());
  emit({ ready: true, version: process.versions.node });
})();
