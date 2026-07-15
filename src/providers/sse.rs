//! Incremental Server-Sent Events framing, shared by streaming adapters.
//! The framing layer is protocol-neutral: it turns raw HTTP chunks into
//! complete `data:` payloads (and the `[DONE]` sentinel) without knowing
//! which provider's JSON they carry.

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SseEvent {
    Data(String),
    Done,
}

/// Incremental Server-Sent Events decoder. It handles LF, CRLF, and CR line
/// endings, comments/other fields, multiple `data:` lines, fragmented UTF-8,
/// and a final event without a trailing blank line. Consumed bytes are
/// compacted once per HTTP chunk rather than drained once per line.
#[derive(Default)]
pub(crate) struct SseDecoder {
    buffer: Vec<u8>,
    data: Vec<u8>,
    has_data: bool,
    first_line: bool,
}

impl SseDecoder {
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, std::string::FromUtf8Error> {
        self.buffer.extend_from_slice(chunk);
        self.parse(false)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<SseEvent>, std::string::FromUtf8Error> {
        self.parse(true)
    }

    fn parse(&mut self, eof: bool) -> Result<Vec<SseEvent>, std::string::FromUtf8Error> {
        let mut events = Vec::new();
        let mut consumed = 0usize;
        loop {
            let mut terminator = None;
            for index in consumed..self.buffer.len() {
                match self.buffer[index] {
                    b'\n' => {
                        terminator = Some((index, index + 1));
                        break;
                    }
                    b'\r' if index + 1 < self.buffer.len() => {
                        terminator = Some((
                            index,
                            index + 1 + usize::from(self.buffer[index + 1] == b'\n'),
                        ));
                        break;
                    }
                    b'\r' if eof => {
                        terminator = Some((index, index + 1));
                        break;
                    }
                    _ => {}
                }
            }

            let Some((line_end, next)) = terminator else {
                if eof && consumed < self.buffer.len() {
                    let line = &self.buffer[consumed..];
                    process_sse_line(
                        line,
                        &mut self.data,
                        &mut self.has_data,
                        &mut self.first_line,
                        &mut events,
                    )?;
                    consumed = self.buffer.len();
                }
                break;
            };
            let line = &self.buffer[consumed..line_end];
            process_sse_line(
                line,
                &mut self.data,
                &mut self.has_data,
                &mut self.first_line,
                &mut events,
            )?;
            consumed = next;
        }

        if consumed == self.buffer.len() {
            self.buffer.clear();
        } else if consumed > 0 {
            self.buffer.drain(..consumed);
        }
        if eof && self.has_data {
            dispatch_sse_data(&mut self.data, &mut self.has_data, &mut events)?;
        }
        Ok(events)
    }
}

fn process_sse_line(
    mut line: &[u8],
    data: &mut Vec<u8>,
    has_data: &mut bool,
    first_line: &mut bool,
    events: &mut Vec<SseEvent>,
) -> Result<(), std::string::FromUtf8Error> {
    if !*first_line {
        *first_line = true;
        line = line.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(line);
    }
    if line.is_empty() {
        if *has_data {
            dispatch_sse_data(data, has_data, events)?;
        }
        return Ok(());
    }
    if line[0] == b':' {
        return Ok(());
    }
    let colon = line
        .iter()
        .position(|byte| *byte == b':')
        .unwrap_or(line.len());
    if &line[..colon] != b"data" {
        return Ok(());
    }
    let mut value = if colon < line.len() {
        &line[colon + 1..]
    } else {
        b""
    };
    if value.first() == Some(&b' ') {
        value = &value[1..];
    }
    if *has_data {
        data.push(b'\n');
    }
    data.extend_from_slice(value);
    *has_data = true;
    Ok(())
}

fn dispatch_sse_data(
    data: &mut Vec<u8>,
    has_data: &mut bool,
    events: &mut Vec<SseEvent>,
) -> Result<(), std::string::FromUtf8Error> {
    let value = String::from_utf8(std::mem::take(data))?;
    *has_data = false;
    if value == "[DONE]" {
        events.push(SseEvent::Done);
    } else {
        events.push(SseEvent::Data(value));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_decoder_handles_fragmented_multiline_events_and_crlf() {
        let mut decoder = SseDecoder::default();
        assert!(
            decoder
                .push(b": keepalive\r\nid: 1\r\nda")
                .unwrap()
                .is_empty()
        );
        assert!(decoder.push(b"ta: {\"value\":\r\n").unwrap().is_empty());
        assert_eq!(
            decoder.push(b"data: 1}\r\n\r\n").unwrap(),
            vec![SseEvent::Data("{\"value\":\n1}".to_string())]
        );
    }
    #[test]
    fn sse_decoder_preserves_split_utf8_and_finishes_unterminated_event() {
        let mut decoder = SseDecoder::default();
        let event = "data: {\"text\":\"hé\"}".as_bytes();
        let split = event.iter().position(|byte| *byte == 0xc3).unwrap() + 1;
        assert!(decoder.push(&event[..split]).unwrap().is_empty());
        assert!(decoder.push(&event[split..]).unwrap().is_empty());
        assert_eq!(
            decoder.finish().unwrap(),
            vec![SseEvent::Data("{\"text\":\"hé\"}".to_string())]
        );
    }
    #[test]
    fn sse_decoder_handles_many_events_done_and_bom() {
        let mut decoder = SseDecoder::default();
        let events = decoder
            .push(b"\xef\xbb\xbfdata: one\n\ndata: two\r\rdata: [DONE]\n\n")
            .unwrap();
        assert_eq!(
            events,
            vec![
                SseEvent::Data("one".to_string()),
                SseEvent::Data("two".to_string()),
                SseEvent::Done,
            ]
        );
    }
}
