//! Decode AWS event-stream envelopes before parsing JSON (never scan binary data for braces).
use serde_json::Value;
use std::collections::HashMap;

const MAX_FRAME: usize = 16 * 1024 * 1024;

pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb88320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

pub struct Frame {
    pub headers: HashMap<String, String>,
    pub payload: Value,
}

pub fn decode(buffer: &mut Vec<u8>) -> Result<Vec<Frame>, String> {
    let mut frames = vec![];
    let mut offset = 0;
    while buffer.len() - offset >= 12 {
        let frame = &buffer[offset..];
        let total = u32::from_be_bytes(frame[0..4].try_into().unwrap()) as usize;
        let header_len = u32::from_be_bytes(frame[4..8].try_into().unwrap()) as usize;
        if !(16..=MAX_FRAME).contains(&total) || header_len > total - 16 {
            return Err("Invalid Kiro event-stream frame length".into());
        }
        if crc32(&frame[..8]) != u32::from_be_bytes(frame[8..12].try_into().unwrap()) {
            return Err("Invalid Kiro event-stream prelude checksum".into());
        }
        if frame.len() < total {
            break;
        }
        if crc32(&frame[..total - 4])
            != u32::from_be_bytes(frame[total - 4..total].try_into().unwrap())
        {
            return Err("Invalid Kiro event-stream checksum".into());
        }
        let mut headers = HashMap::new();
        let header = &frame[12..12 + header_len];
        let mut cursor = 0;
        while cursor < header.len() {
            let size = header[cursor] as usize;
            cursor += 1;
            if cursor + size + 1 > header.len() {
                return Err("Malformed Kiro event headers".into());
            }
            let name = std::str::from_utf8(&header[cursor..cursor + size])
                .map_err(|_| "Invalid Kiro event header name")?
                .to_owned();
            cursor += size;
            let kind = header[cursor];
            cursor += 1;
            let size = match kind {
                0 | 1 => 0,
                2 => 1,
                3 => 2,
                4 => 4,
                5 | 8 => 8,
                9 => 16,
                6 | 7 => {
                    if cursor + 2 > header.len() {
                        return Err("Truncated Kiro event header".into());
                    }
                    let size =
                        u16::from_be_bytes(header[cursor..cursor + 2].try_into().unwrap()) as usize;
                    cursor += 2;
                    size
                }
                _ => return Err("Unknown Kiro event header type".into()),
            };
            if cursor + size > header.len() {
                return Err("Truncated Kiro event header value".into());
            }
            if kind == 7 {
                headers.insert(
                    name,
                    std::str::from_utf8(&header[cursor..cursor + size])
                        .map_err(|_| "Invalid Kiro event header value")?
                        .to_owned(),
                );
            }
            cursor += size;
        }
        let payload = serde_json::from_slice(&frame[12 + header_len..total - 4])
            .map_err(|_| "Invalid JSON in Kiro event")?;
        frames.push(Frame { headers, payload });
        offset += total;
    }
    buffer.drain(..offset);
    Ok(frames)
}
