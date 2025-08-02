# ----------------------------
# 1. 기반 이미지 설정 (라즈베리파이 64비트 아키텍처용)
# ----------------------------
FROM arm64v8/debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------
# 2. 기본 패키지 및 라즈베리파이 공식 저장소 추가
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    lsb-release wget curl gnupg2 sudo apt-transport-https \
    && rm -rf /var/lib/apt/lists/*

# 라즈베리파이 공식 APT 저장소 추가
RUN curl -sSL http://archive.raspberrypi.com/debian/raspberrypi.gpg.key | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.com/debian/ bookworm main" > /etc/apt/sources.list.d/raspberrypi.list

# ----------------------------
# 3. GStreamer, Python 및 Hailo 관련 패키지 설치
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-setuptools \
    build-essential cmake git \
    libusb-1.0-0 libusb-1.0-0-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x libzmq3-dev \
    libcairo2-dev x11-utils ffmpeg \
    # hailo-all 설치 (의존성 패키지 포함)
    hailo-all \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 4. Hailo SDK에 필요한 Python 패키지 설치
# --break-system-packages 옵션을 추가하여 에러 해결
# ----------------------------
RUN pip3 install --no-cache-dir --break-system-packages numpy pyyaml opencv-python-headless onnx matplotlib

# ----------------------------
# 5. 환경 변수 설정
# ----------------------------
ENV PATH="${PATH}:/usr/bin/hailo"
ENV LD_LIBRARY_PATH="/usr/lib/hailo:${LD_LIBRARY_PATH}"

# ----------------------------
# 6. 컨테이너 실행 시 작업 디렉토리 설정 및 쉘 실행
# ----------------------------
WORKDIR /root
CMD ["bash"]
