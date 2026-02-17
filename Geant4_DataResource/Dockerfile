# 1. Use a clean, standard Ubuntu base
FROM ubuntu:22.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# 2. Install Build Tools (C++, CMake, Expat, Xerces)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    libexpat1-dev \
    libxerces-c-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Download & Compile Geant4 Source (The "Heavy Lifting")
# We use /opt to store the software
WORKDIR /opt

# Download Geant4 11.1 (Stable version)
RUN wget -q https://geant4-data.web.cern.ch/releases/geant4-v11.1.0.tar.gz && \
    tar -xzf geant4-v11.1.0.tar.gz && \
    mkdir geant4-build && \
    cd geant4-build && \
    # Configure CMake (Minimal Install for Server Mode)
    cmake -DCMAKE_INSTALL_PREFIX=/opt/geant4-install \
          -DGEANT4_INSTALL_DATA=ON \
          -DGEANT4_USE_OPENGL_X11=OFF \
          -DGEANT4_USE_QT=OFF \
          -DGEANT4_BUILD_MULTITHREADED=ON \
          ../geant4-v11.1.0 && \
    # Compile (Uses all CPU cores)
    make -j$(nproc) && \
    make install && \
    # Cleanup to keep image smaller
    cd .. && rm -rf geant4-build geant4-v11.1.0 geant4-v11.1.0.tar.gz

# 4. Set Environment Variables so your app finds Geant4
ENV Geant4_DIR=/opt/geant4-install/lib/Geant4-11.1.0/
# 5. Compile YOUR Simulation
WORKDIR /app
COPY . .

# Build your app
RUN mkdir -p build && cd build && \
    # CRITICAL: Load Geant4 variables before compiling
    . /opt/geant4-install/bin/geant4.sh && \
    cmake .. && \
    make -j$(nproc)

# 6. Run the Simulation
WORKDIR /app/build

# CRITICAL FIX: Source the script EVERY TIME the container starts
# This adds the libraries to the path so "./exampleB4a" can find them.
CMD ["/bin/bash", "-c", "source /opt/geant4-install/bin/geant4.sh && ./exampleB4a"]