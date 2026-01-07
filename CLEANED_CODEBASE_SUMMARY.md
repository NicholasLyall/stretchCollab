# Stretch AI 2.0 Cleaned Codebase

This directory contains a cleaned version of the Stretch AI codebase with only the essential files needed to run the PERSISTEDMap_360mapping.py application and its dependencies.

## What Was Copied

### Core Infrastructure (15 files)
- `src/stretch/core/` - Complete core infrastructure including robot, client, interfaces, parameters, task, etc.

### Agent System (25+ files)
- `src/stretch/agent/robot_agent.py` - Main robot agent controller
- `src/stretch/agent/zmq_client.py` - ZMQ communication client
- `src/stretch/agent/operations/` - All operations including enhanced navigation, grasp, pickup, etc.
- `src/stretch/agent/task/` - All task modules including pickup, navigation, emote, etc.

### Mapping & Navigation (10+ files)  
- `src/stretch/mapping/voxel/voxel_map.py` - Core voxel mapping (with navigation bug fix)
- `src/stretch/mapping/` - Complete mapping infrastructure (grid, instance, scene_graph)
- `src/stretch/motion/` - Complete motion planning and control system

### Perception (12 files)
- `src/stretch/perception/wrapper.py` - Perception wrapper
- `src/stretch/perception/detection/` - Object detection modules (OWL, SAM2, YOLO)
- `src/stretch/perception/captioners/` - Vision-language models
- `src/stretch/perception/encoders/` - Feature encoders

### Utilities & Support
- `src/stretch/utils/` - Complete utilities package
- `src/stretch/llms/` - Language model clients and prompts  
- `src/stretch/audio/` - Audio processing and TTS
- `src/stretch/visualization/` - Rerun visualization system

### Configuration
- `src/stretch/config/` - All configuration files including:
  - `default_planner.yaml` - Motion planning configuration
  - `example_cat_map.json` - Object category mappings
  - `urdf/stretch.urdf` - Robot description

### Main Application
- `src/stretch/app/PERSISTEDMap_360mapping.py` - The main unified mapping + manipulation service

### Setup & Data Files
- `pyproject.toml` - Python project configuration
- `src/setup.py` - Package setup script
- `install.sh` - Installation script
- `data/scenes/` - Default simulation scenes
- `yolov8n.pt` and `yoloe-v8l-seg.pt` - YOLO model weights
- `LICENSE` and `META_LICENSE` - License files

## Statistics
- **199 Python files** copied
- **114MB** total size
- **Complete dependency chain** for PERSISTEDMap_360mapping.py

## What Was Excluded
- Experimental apps and demos
- Test files and test data  
- Documentation files (docs/)
- Docker configuration
- Third-party submodules
- Unused legacy code
- Development tools and scripts

## Key Features Preserved
- ✅ **Unified Mapping + Manipulation Service** - Full PERSISTEDMap_360mapping.py functionality
- ✅ **Grab Command Integration** - Object pickup and placement
- ✅ **Navigation Bug Fix** - Fixed coordinate system bug in voxel_map.py
- ✅ **Complete Perception Pipeline** - YOLO, OWL-ViT, SAM2 object detection
- ✅ **Robust Motion Planning** - A* pathfinding with obstacle avoidance
- ✅ **Rerun Visualization** - 3D scene visualization and mapping
- ✅ **Audio Support** - Text-to-speech and audio processing
- ✅ **LLM Integration** - Language model clients for natural interaction

## Usage
To run the main application:
```bash
cd stretch_ai_2.0_cleaned
python -m stretch.app.PERSISTEDMap_360mapping --robot_ip YOUR_ROBOT_IP
```

This cleaned codebase provides all the functionality needed for spatial mapping, object detection, navigation, and manipulation tasks while removing unnecessary complexity and experimental code.