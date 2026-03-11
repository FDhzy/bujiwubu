# Building Android APK for Blind Navigation Assistant

## Prerequisites

Building an Android APK requires a Linux environment. On Windows, the recommended approach is using WSL2 (Windows Subsystem for Linux).

## Method 1: Using WSL2 (Recommended for Windows)

### Step 1: Install WSL2

Open PowerShell as Administrator and run:

```powershell
wsl --install -d Ubuntu
```

Restart your computer if prompted.

### Step 2: Set up the build environment in WSL2

Open Ubuntu from the Start menu and run these commands:

```bash
# Update system
sudo apt update
sudo apt upgrade -y

# Install required dependencies
sudo apt install -y python3-pip build-essential git zip unzip \
    openjdk-17-jdk autoconf libtool pkg-config zlib1g-dev \
    libncurses5-dev libncursesw5-dev libtinfo5 cmake \
    libffi-dev libssl-dev

# Install Buildozer and Cython
pip3 install --upgrade buildozer cython

# Add pip binaries to PATH (add this to ~/.bashrc for persistence)
export PATH=$PATH:~/.local/bin
```

### Step 3: Navigate to your project

```bash
# Access your Windows drive (adjust path as needed)
cd "/mnt/d/ -- 不积洼步_1.23/不积洼步_1.23/不积洼步(1)/不积洼步"
```

### Step 4: Build the APK

```bash
# Build debug APK (faster, for testing)
buildozer android debug

# Or build release APK (optimized, for distribution)
buildozer android release
```

**Note:** The first build will take 30-60 minutes as it downloads Android SDK, NDK, and compiles dependencies.

### Step 5: Find your APK

After successful build, the APK will be located at:
```
./bin/blindnavassist-1.0.0-arm64-v8a_armeabi-v7a-debug.apk
```

Transfer this file to your Android phone and install it.

---

## Method 2: Using Docker

### Step 1: Install Docker Desktop for Windows

Download and install from: https://www.docker.com/products/docker-desktop

### Step 2: Pull Buildozer Docker image

```powershell
docker pull kivy/buildozer
```

### Step 3: Build APK using Docker

```powershell
cd "d:\ -- 不积洼步_1.23\不积洼步_1.23\不积洼步(1)\不积洼步"

docker run --rm -v ${PWD}:/home/user/hostcwd kivy/buildozer android debug
```

---

## Method 3: Using GitHub Actions (Cloud Build)

Create a `.github/workflows/build.yml` file in your repository:

```yaml
name: Build Android APK

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install buildozer cython
          sudo apt update
          sudo apt install -y openjdk-17-jdk
      
      - name: Build APK
        run: buildozer android debug
      
      - name: Upload APK
        uses: actions/upload-artifact@v3
        with:
          name: android-apk
          path: bin/*.apk
```

Push to GitHub and download the APK from Actions artifacts.

---

## Troubleshooting

### Common Issues:

1. **"buildozer: command not found"**
   - Solution: Add `~/.local/bin` to PATH: `export PATH=$PATH:~/.local/bin`

2. **"Permission denied" errors**
   - Solution: Run `chmod +x ~/.local/bin/buildozer`

3. **Build fails with Java errors**
   - Solution: Ensure Java 17 is installed: `java -version`

4. **Out of disk space**
   - Solution: Free up at least 10GB of space for Android SDK/NDK

5. **Slow build on WSL2**
   - Solution: Move project to WSL2 filesystem (e.g., `~/projects/`) instead of accessing Windows drive

### Optimization Tips:

- **First build**: ~30-60 minutes (downloads SDK, NDK, dependencies)
- **Subsequent builds**: ~5-15 minutes (only recompiles changed code)
- Use `buildozer android debug` for faster testing builds
- Use `buildozer android release` for production APKs

---

## Installing APK on Android Phone

1. Transfer the APK file to your Android phone via USB, email, or cloud storage
2. Open the APK file on your phone
3. If prompted, enable "Install from unknown sources" in Settings
4. Follow the installation prompts
5. Grant required permissions (Camera, Microphone, Vibration) when prompted

---

## Build Configuration

The build is configured in `buildozer.spec`:
- **Package name**: org.accesstech.blindnavassist
- **Version**: 1.0.0
- **Min Android API**: 26 (Android 8.0)
- **Target Android API**: 33 (Android 13)
- **Architectures**: arm64-v8a, armeabi-v7a (covers most devices)
- **Permissions**: CAMERA, VIBRATE, WAKE_LOCK, RECORD_AUDIO

---

## Quick Start (WSL2)

```bash
# One-time setup
wsl --install -d Ubuntu
# (Restart computer, then open Ubuntu)

sudo apt update && sudo apt install -y python3-pip build-essential git openjdk-17-jdk
pip3 install buildozer cython
export PATH=$PATH:~/.local/bin

# Build APK
cd "/mnt/d/ -- 不积洼步_1.23/不积洼步_1.23/不积洼步(1)/不积洼步"
buildozer android debug

# APK will be in: ./bin/
```

---

For more information, visit:
- Buildozer Documentation: https://buildozer.readthedocs.io/
- Kivy Android Documentation: https://kivy.org/doc/stable/guide/packaging-android.html
