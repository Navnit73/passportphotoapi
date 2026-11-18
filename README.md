# passportphotoapi

## US Passport routes (`/us`, standalone engine)

Separate engine in `us_passport/`, mounted in the main app with its own measurement, alignment, crop and compliance logic
(shares only the background-removal service and the JPEG size encoder).

```bash
python main.py   # US routes at /us/process, /us/validate
```

| Endpoint | Description |
|----------|-------------|
| `POST /us/process` (X-API-Key) | Same response schema as the main `/process`: 600x600 JPEG ≤240KB, head 50–69%, eyes 56–69% from bottom |
| `POST /us/validate` (X-API-Key) | Same schema as the main `/validate`, judged against US rules |
| `GET /us/requirements` | Rules in use |

Pipeline: face + iris landmarks → level the eye line → MODNet matte (hair top) → white background →
solve crop (head 60%, eyes 62.5%, centred) → encode → **re-measure the encoded file** and re-plan if any rule fails.
Tests: `pytest tests/test_us_passport.py`.
