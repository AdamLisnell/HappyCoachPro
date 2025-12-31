# HappyCoach2 🏌️‍♂️

AI-powered golf swing analyzer with real-time pose detection and biomechanics feedback.

## Features

- 📹 **Real-time pose detection** using MediaPipe
- 🦴 **Skeleton overlay** that tracks body movements
- 📊 **Swing analysis** with phase detection (Address → Top → Impact → Finish)
- 📐 **Angle calculations** for spine, elbow, knee, hip rotation
- 💡 **Coaching feedback** based on biomechanics

## Tech Stack

### Backend (Python)
- **FastAPI** - Modern async API framework
- **MediaPipe** - ML-based pose detection
- **OpenCV** - Video/image processing
- **WebSocket** - Real-time communication

### Frontend (React)
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Canvas API** - Skeleton rendering
- **PWA** - Installable on mobile

## Architecture
```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│    Frontend     │◄──────────────────►│     Backend      │
│  (React PWA)    │     (frames)       │    (FastAPI)     │
│                 │                    │                  │
│  • Camera       │                    │  • MediaPipe     │
│  • Skeleton UI  │◄───────────────────│  • Analysis      │
│  • Results      │    (landmarks)     │  • Scoring       │
└─────────────────┘                    └──────────────────┘
```

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Webcam or mobile camera

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Project Structure
```
HappyCoach2/
├── backend/
│   ├── core/
│   │   ├── domain/         # Data models
│   │   ├── services/       # Business logic
│   │   └── use_cases/      # Application logic
│   ├── api/                # FastAPI routes & WebSocket
│   ├── main.py             # Entry point
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── hooks/          # Custom hooks
│   │   └── styles/         # CSS
│   └── package.json
└── docker-compose.yml
```

## License

MIT

## Author

Adam Lisnell - [Lund University](https://lu.se) → [KTH MSc Application](https://kth.se)