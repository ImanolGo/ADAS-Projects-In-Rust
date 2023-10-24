# 🚦 Coding Activity 5 – Street Light Detection and Recognition for ADAS 🚦

## 📝 Description

An upcoming trend in ADAS technologies focuses on the detection and recognition of street lights. This is incredibly useful as drivers may sometimes miss or misinterpret traffic lights for various reasons.

## 🎯 Problem Statement

- Develop the Street Light Detection system in Rust.
- There are two approaches for this problem:
  1. **YOLOv3 Approach**: Use the YOLOv3 algorithm as in the previous coding activity. Filter out all object classes to focus solely on street lights. Further, apply a color detection algorithm to determine which color the street lamp is currently displaying (RED, AMBER, GREEN) or if it is DEACTIVE.
  2. **Direct Deep Learning Approach**: Use a dataset of various street light statuses (if available) to train a CNN-based Deep Neural Network. This model should directly output one of the four classes: RED, AMBER, GREEN, or DEACTIVE.
- Test the developed system on random road images and/or video feed.

## 🛠 Technologies

- **Programming Language**: Rust
- **Object Detection Algorithm**: YOLOv3 (optional)
- **Color Recognition Algorithm**: [Specify Algorithm if applicable]

## 🚫 No References

This activity expects the implementation to be carried out without direct help or direct code. However, students are encouraged to refer to online materials to learn and understand relevant concepts and algorithms.

## 🏁 Getting Started

[Provide setup and running instructions here]
