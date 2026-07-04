# Sample Videos

This folder documents one small demo clip for users who do not have a soccer video ready:

- `default_soccer_clip.mp4`

In the Gradio UI, upload your own clip or select **Use sample soccer clip** before clicking **Analyze Video**. Uploaded videos always take priority over the sample. The default sample path uses cached outputs from `sample_outputs/` for speed on Hugging Face CPU Spaces; select **Reprocess sample with YOLO** only when you want to run the full detector/tracker on this sample clip.

The sample video is intentionally **not committed as a normal Git blob**. For Hugging Face Spaces, upload it separately with Hugging Face Hub/Xet-compatible storage at this exact path:

```bash
hf upload abhinavvathadi/gamelens-ai sample_videos/default_soccer_clip.mp4 sample_videos/default_soccer_clip.mp4 --repo-type space
```

The app uses the same video path locally and on the Space: `sample_videos/default_soccer_clip.mp4`. Cached sample metrics, charts, report text, and downloads live in `sample_outputs/` and are committed as lightweight Git files.

## Adding Local Clips

You can keep local testing clips here, but `.mp4` files are ignored by `.gitignore` so they are not pushed accidentally through normal Git.

- Recommended length: **30-90 seconds**
- Format: `.mp4`
- Camera: broadcast-style or wide-angle footage where players are clearly visible

## Note on Git

Do not add `default_soccer_clip.mp4` or other videos with normal `git add`. Keep `sample_videos/README.md` in Git, and upload the MP4 separately to Hugging Face Space storage.

## Trimming Clips

```bash
ffmpeg -i input.mp4 -ss 00:00:00 -t 00:00:60 -c copy sample_soccer_clip.mp4
```