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
# 3. GStreamer, Python 및 빌드 관련 의존성 패키지 설치
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-setuptools \
    build-essential cmake git binutils \
    libusb-1.0-0 libusb-1.0-0-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x libzmq3-dev \
    libcairo2-dev x11-utils ffmpeg \
    dpkg-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------
# 4. HailoRT 4.22.0 C++ 라이브러리 및 Python 모듈(.whl) 설치
# ----------------------------
# Dockerfile과 같은 경로에 'hailort_4.22.0_arm64.deb'와
# 'hailort-4.22.0-cp311-cp311-linux_aarch64.whl' 파일이 있어야 합니다.
COPY hailort_4.22.0_arm64.deb .
COPY hailort-4.22.0-cp311-cp311-linux_aarch64.whl .

# DEB 파일의 내용을 수동으로 추출하여 필요한 파일만 복사
RUN mkdir -p /tmp/deb_extract && \
    dpkg-deb -R hailort_4.22.0_arm64.deb /tmp/deb_extract && \
    cp -r /tmp/deb_extract/usr/* /usr/ && \
    rm -rf /tmp/deb_extract hailort_4.22.0_arm64.deb

# Python 모듈을 pip으로 설치 (시스템 패키지 관리자 오류를 무시)
RUN pip3 install --break-system-packages hailort-4.22.0-cp311-cp311-linux_aarch64.whl && \
    rm hailort-4.22.0-cp311-cp311-linux_aarch64.whl

# ----------------------------
# 5. 기타 Python 패키지 설치
# ----------------------------
RUN pip3 install --no-cache-dir --break-system-packages numpy pyyaml opencv-python-headless onnx matplotlib

# ----------------------------
# 6. 환경 변수 설정
# ----------------------------
ENV PATH="${PATH}:/usr/bin/hailo"
ENV LD_LIBRARY_PATH="/usr/lib"

# ----------------------------
# 7. 컨테이너 실행 시 작업 디렉토리 설정 및 쉘 실행
# ----------------------------
WORKDIR /root
CMD ["bash"]
