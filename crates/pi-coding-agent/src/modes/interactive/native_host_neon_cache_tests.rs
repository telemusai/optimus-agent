use super::*;
use pi_ai::types::{Message, UserContent, UserMessage};

fn transcript() -> Transcript {
    let mut mode = super::super::tests::stash_mode("neon-cache-fixture");
    crate::modes::interactive::theme::theme::init_theme(Some("neon"), false);
    mode.fullscreen_enabled = true;
    Transcript::new(Rc::new(RefCell::new(mode)))
}

#[test]
fn long_transcript_editor_repaints_do_not_reformat_unchanged_neon_lines() {
    let mut transcript = transcript();
    let text = (0..1000)
        .map(|i| format!("Row {i}: café 世界 👩‍💻 {}", "output ".repeat(8)))
        .collect::<Vec<_>>()
        .join("\n");
    transcript.message(AgentMessage::Message(Message::User(UserMessage::new(
        UserContent::Text(text), 1_790_200_000_000,
    ))), false);
    let started = Instant::now();
    let expected = transcript.render(150.0);
    let cold = started.elapsed();
    assert!(expected.len() >= 1000);
    LINE_FORMATS.with(|count| count.set(0));
    let started = Instant::now();
    for _ in 0..5 {
        assert_eq!(transcript.render(150.0), expected);
    }
    eprintln!("neon 1000-line transcript: cold={cold:?}, warm mean={:?}", started.elapsed() / 5);
    assert_eq!(LINE_FORMATS.with(Cell::get), 0, "editor/status repaints reformatted unchanged history");
}

fn check_frame(
    cache: &mut TimelineCache,
    timeline: Timeline,
    rows: &[(String, Option<RowMeta>, bool)],
    expected_formats: usize,
) {
    let expected = rows.iter().map(|(text, meta, first)| timeline.line(text, meta.as_ref(), *first)).collect::<Vec<_>>();
    let expected_columns = rows.iter().map(|(text, _, _)| Some((timeline.left,
        timeline.left + visible_width(strip_ansi(text).trim_end()).min(timeline.content_width())))).collect::<Vec<_>>();
    LINE_FORMATS.with(|count| count.set(0));
    let (actual, columns) = cache.render(timeline, rows.iter().map(|(text, meta, first)| (text.as_str(), meta.as_ref(), *first)));
    assert_eq!(actual, expected);
    assert_eq!(columns, expected_columns);
    assert_eq!(LINE_FORMATS.with(Cell::get), expected_formats);
    assert_eq!(cache.lines.len(), rows.len());
}

#[test]
fn timeline_cache_updates_only_changed_lines_and_releases_removed_history() {
    let _transcript = transcript();
    let mut cache = TimelineCache::default();
    let timeline = Timeline::new(120);
    let mut rows = vec![("first 世界".into(), None, false), ("streaming".into(), None, false)];
    check_frame(&mut cache, timeline, &rows, 2);
    check_frame(&mut cache, timeline, &rows, 0);
    rows[1].0.push_str(" delta 👩‍💻");
    check_frame(&mut cache, timeline, &rows, 1);
    rows.push(("appended".into(), None, false));
    check_frame(&mut cache, timeline, &rows, 1);
    rows.insert(0, ("older page".into(), None, false));
    check_frame(&mut cache, timeline, &rows, 4);
    rows.truncate(1);
    check_frame(&mut cache, timeline, &rows, 0);
    rows.clear();
    check_frame(&mut cache, timeline, &rows, 0);
}

#[test]
fn timeline_cache_refreshes_tool_status_timestamps_elapsed_time_and_anchors() {
    let _transcript = transcript();
    let mut cache = TimelineCache::default();
    let timeline = Timeline::new(120);
    let mut rows = vec![("tool result".into(), Some(RowMeta::new(Kind::Tool, Some(1_790_200_000_000))), true)];
    check_frame(&mut cache, timeline, &rows, 1);
    rows[0].1.as_mut().unwrap().kind = Kind::Success;
    check_frame(&mut cache, timeline, &rows, 1);
    rows[0].1.as_mut().unwrap().elapsed = Some(Duration::from_millis(1234));
    check_frame(&mut cache, timeline, &rows, 1);
    rows[0].1.as_mut().unwrap().timestamp = Some(1_790_200_005_000);
    check_frame(&mut cache, timeline, &rows, 1);
    rows[0].2 = false;
    check_frame(&mut cache, timeline, &rows, 1);
    rows[0].1 = None;
    check_frame(&mut cache, timeline, &rows, 1);
    check_frame(&mut cache, timeline, &rows, 0);
}

#[test]
fn timeline_cache_refreshes_resize_theme_reload_and_explicit_invalidation() {
    let _transcript = transcript();
    let mut cache = TimelineCache::default();
    let rows = vec![("\x1b[31mstyled 世界\x1b[0m".into(), None, false)];
    for width in [120, 40, 24, 150] {
        check_frame(&mut cache, Timeline::new(width), &rows, 1);
        check_frame(&mut cache, Timeline::new(width), &rows, 0);
    }
    let timeline = Timeline::new(150);
    // A new instance of the same named theme must also invalidate the cache.
    crate::modes::interactive::theme::theme::set_theme_instance((*theme()).clone());
    check_frame(&mut cache, timeline, &rows, 1);
    crate::modes::interactive::theme::theme::init_theme(Some("light"), false);
    check_frame(&mut cache, timeline, &rows, 1);
    cache.clear();
    assert!(cache.lines.is_empty());
    assert!(cache.palette.is_none());
    check_frame(&mut cache, timeline, &rows, 1);
}

#[test]
fn timeline_cache_preserves_unicode_links_and_graphics_on_repaint() {
    let _transcript = transcript();
    let mut cache = TimelineCache::default();
    let graphic = pi_tui::terminal_image::position_image("\x1bP0;1;0q~\x1b\\", 3);
    let rows = vec![
        ("\x1b]8;;https://example.test\x07café 世界 👩‍💻\x1b]8;;\x07".into(), None, false),
        (graphic, None, false),
    ];
    for width in [120, 40, 24] {
        check_frame(&mut cache, Timeline::new(width), &rows, 3);
        check_frame(&mut cache, Timeline::new(width), &rows, 0);
        assert_eq!(pi_tui::terminal_image::image_row_count(&cache.lines[1].rendered), Some(3));
    }
}

#[test]
fn transcript_cache_clears_on_replacement_invalidation_and_leaving_neon() {
    let mut transcript = transcript();
    transcript.panel("old transcript");
    transcript.render(120.0);
    assert!(!transcript.timeline_cache.lines.is_empty());
    transcript.invalidate();
    assert!(transcript.timeline_cache.lines.is_empty());
    transcript.render(120.0);
    transcript.replace(Vec::new());
    assert!(transcript.timeline_cache.lines.is_empty());
    transcript.panel("new transcript");
    let expected = transcript.render(120.0);
    assert!(!expected.join("\n").contains("old transcript"));
    transcript.mode.borrow_mut().fullscreen_enabled = false;
    transcript.render(120.0);
    assert!(transcript.timeline_cache.lines.is_empty());
    assert!(transcript.get_selection_columns().is_empty());
    transcript.mode.borrow_mut().fullscreen_enabled = true;
    assert_eq!(transcript.render(120.0), expected);
}
