# TreeTrack AI — AI Disease Detection Module

Production-grade, modular plant disease detection backend service supporting 5 crops:
1. **Tomato** (`ixrbhii/tomato-disease-convnext` - 13 classes)
2. **Chilli** (`NiroshanDb23/chili-disease-vgg16` - 6 classes)
3. **Brinjal** (`ixrbhii/brinjal-disease-convnext` - 8 classes)
4. **Ladies' Finger (Okra)** (`ixrbhii/okra-disease-convnext` - 6 classes)
5. **Cucumber** (`ixrbhii/cucumber-disease-convnext` - 8 classes)

---

## Architecture & Features

- **FastAPI Backend**: Located entirely inside `Backend/Disease`.
- **Image Validation Service** (`app/services/image_service.py`):
  - Validates file signatures (JPEG, PNG, WebP) and checks corruption.
  - Corrects EXIF smartphone camera orientation.
  - Normalizes tensors using ImageNet / ConvNeXt specifications.
- **Model Registry & Service** (`app/services/model_service.py`):
  - Maps crops to Hugging Face models.
  - Caches loaded weights in memory with non-blocking fallback inference.
- **Treatment & Prevention Service** (`app/services/treatment_service.py`):
  - Backed by verified Indian agricultural research standards (**ICAR**, **IIHR**, **TNAU**).
  - Explicitly separates non-chemical cultural controls from chemical remedies.
  - Flags predictions as preliminary and warns when human review is required.
- **Interactive Web Frontend** (`Frountend/templates/`):
  - Connected directly to `/api/v1/disease/analyze`.
  - Live leaf photo upload, instant progress indicator, dynamic diagnosis card rendering.

---

## API Endpoints (`/api/v1`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status |
| `GET` | `/api/v1/disease/crops` | List all 5 supported crops and model metadata |
| `GET` | `/api/v1/disease/crops/{crop}/classes` | Get verified disease labels for a specific crop |
| `POST` | `/api/v1/disease/analyze` | Multipart upload (`crop`: string, `image`: file) |
| `GET` | `/` | Web frontend landing page |
| `GET` | `/chat` | Web frontend disease detection chat |

---

## Quick Start (PowerShell)

### 1. Run the FastAPI Backend Server

```powershell
# From project root:
.\env\Scripts\Activate.ps1
cd Backend
uvicorn main:app --reload --port 8000
```

### 2. Open the Connected Web App

- **Landing Page**: [http://localhost:8000/](http://localhost:8000/)
- **Disease Detection Chat**: [http://localhost:8000/chat](http://localhost:8000/chat)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Running the Automated Test Suite

```powershell
# From Backend directory:
..\env\Scripts\python.exe -m pytest -v
```

All 13 unit and integration tests validate health status, file validation (format, size limits, corrupted bytes), crop routes, and end-to-end leaf prediction schema.
