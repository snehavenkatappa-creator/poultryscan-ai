---
title: PoultryScan AI
emoji: 🐔
colorFrom: green
colorTo: emerald
sdk: docker
pinned: false
app_port: 7860
fullWidth: true
---

# PoultryScan AI — Poultry Disease Detection System

A production Flask web application for AI-powered poultry disease detection using MobileNetV2 transfer learning.

## Features
- 5-Class disease classification: Coccidiosis, Fowl Pox, Healthy, Newcastle Disease, Salmonella
- Multi-chicken flock analysis with bounding box detection
- Comprehensive scan history with SQLite persistence
- Disease information encyclopedia
- Real-time analytics dashboard

## Technical Stack
- **Backend**: Flask + Gunicorn (Python 3.11)
- **ML**: TensorFlow/Keras (MobileNetV2) + PyTorch (FasterRCNN-MobileNetV3)
- **DB**: SQLite
- **Container**: Docker (CPU-only inference)
