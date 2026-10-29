FROM debian:stable-slim

# libhdf5-dev: h5py has no wheels for the newest Python, so tox builds it from source
# there. The other Python versions tox tests are downloaded by uv (tox-uv).

RUN apt update && \
apt-get install -y \
chromium \
cmake \
g++ \
gfortran \
git \
libblas-dev \
libhdf5-dev \
liblapack-dev \
libopenmpi-dev \
pkg-config \
python3 \
python3-build \
python3-pip \
python3-venv \
twine
