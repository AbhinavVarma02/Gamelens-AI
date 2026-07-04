# Sample Outputs

Cached sample outputs used by Hugging Face Spaces for a fast CPU-friendly demo.

These files let the `Use sample soccer clip` option return quickly without running YOLO + ByteTrack by default:

- `sample_metrics.json`
- `sample_tracking_data.csv`
- `sample_chart_zone_activity.png`
- `sample_chart_top_track_segments.png`
- `sample_chart_player_count.png`
- `sample_chart_heatmap.png`
- `sample_report.txt`

Uploaded videos still run the real YOLO + ByteTrack pipeline. The sample MP4 itself remains at `sample_videos/default_soccer_clip.mp4` and is uploaded to Hugging Face Hub/Xet storage separately, not committed as a normal Git blob.