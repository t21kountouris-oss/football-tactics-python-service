# Football Tactics Python Inference Service

ONNX-based offline inference for field detection, player tracking, and ball detection.

## Deploy to Railway

1. Push this folder to GitHub
2. Railway → Deploy from GitHub repo
3. Add environment variables (R2 credentials)
4. Done!

## Environment Variables (Add in Railway)

```
R2_ENDPOINT_URL=https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_key
R2_SECRET_ACCESS_KEY=your_secret
R2_BUCKET_NAME=football-tactics-models
PORT=5000
```

## After Deployment

Get Railway URL → Add to Supabase as `PYTHON_SERVICE_URL`
