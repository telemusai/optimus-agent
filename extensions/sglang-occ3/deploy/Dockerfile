# Additive OCC-3 native development layer. Build context: extensions/sglang-occ3.
# Keep the exact working container image, SGLang, weights and serving options.
# The V3 controller reconstructs the original container configuration
# explicitly; the WORKDIR below only stages build inputs and is not a serving
# configuration change.
#
# BASE_IMAGE is required and has no default: resolve the original container's
# full .Image ID and pass a verified local tag pointing to that ID. A Docker
# image ID is not a registry manifest digest; never fabricate a registry@sha256
# reference from it. The root operator verifies the local base tag/full ID
# before and after the build, and builds with --pull=false --network=none.
# Do not use the legacy deploy.sh/occ3_controller.py paths for a live swap.
#
# This layer stages the reviewed plugin sources, exactly the three reviewed
# source manifests and the root-staged baseline/profile inputs, applies the
# pinned SGLang source overlay through deploy.v3.native_package, installs the
# plugin package offline (no index, no build isolation, no cache, no network
# fallback) and verifies the installed result in an isolated interpreter.
# It is not a verified native latent handoff and not serving-readiness proof.
# No weights, model dependencies, tuning or entrypoint are replaced here.

# Required: the exact image identity of the preserved working container.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

# No LABEL instructions: the derived image must add no metadata beyond the
# plugin layer, so a container from it is label-identical to the original
# (the original carries an ops-added neo.fp8-adaptive-update label which the
# controller replicates via --label).

# The base image runs SGLang from a venv at /opt/sglang (its PATH is
# inherited), so plain `python3` below is the same interpreter that runs
# the server; the plugin must land in that environment to be discovered.
WORKDIR /opt/occ3-bridge

# Build context is the sglang-occ3 extension root (the parent of deploy/):
# ../occ3_sglang and ../pyproject.toml relative to this file.
# README.md is referenced by pyproject.toml; the glob keeps the build
# working while it is absent from the tree (a placeholder is written in
# its place below -- metadata only, the package is unaffected).
COPY occ3_sglang/ ./occ3_sglang/
COPY pyproject.toml README.md* ./
# CPU-only readability probe plus the offline build/packaging utilities.
# Only these deploy/v3 files are staged; nothing else from deploy/ is used.
COPY deploy/v3/artifact_preflight.py deploy/v3/__init__.py deploy/v3/source_patch.py deploy/v3/native_package.py deploy/v3/native_image_build.py ./deploy/v3/
# Exactly the three reviewed source manifests, by name, from patches/.
COPY patches/capture-eager-mtp.v1.json patches/native-startup.v2.json patches/scheduler-cache-ipc.v1.json ./native-build-inputs/
# Root-staged packaging inputs: the root operator stages these two files
# into deploy/native-build-inputs/ before the build. Their absence fails
# the build; there are no fallback inputs and no generated baselines.
COPY deploy/native-build-inputs/source-baseline-expanded.json deploy/native-build-inputs/native-profile-input.json ./native-build-inputs/

# The only build selection switch. Default false; the driver accepts only the
# exact strings "true"/"false" and fails the build on anything else.
ARG OCC3_EXPERIMENTAL_STARTUP=false

# One exclusive disposable layer: staged-input verification, the pinned
# source overlay, the offline package install and the isolated
# installed-content proof. Missing build tools or any digest, membership or
# byte mismatch fails the build; there is no network fallback and no
# partial-apply resume.
RUN if [ ! -f README.md ]; then \
        printf '# occ3-bridge\n\nExperimental SGLang connector for the generic Optimus OCC-3 bridge.\n' > README.md; \
    fi && \
    python3 -m deploy.v3.native_image_build \
    --base-image "${BASE_IMAGE}" \
    --sglang-srt /sgl-workspace/sglang/python/sglang/srt \
    --occ3-sglang /opt/occ3-bridge/occ3_sglang \
    --staged-inputs /opt/occ3-bridge/native-build-inputs \
    --experimental-startup "${OCC3_EXPERIMENTAL_STARTUP}" \
    --disposable-tree
