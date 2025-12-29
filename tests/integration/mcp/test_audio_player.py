#!/usr/bin/env python3
"""
Simple test for audio player detection
"""

import platform
import shutil


def get_audio_player_command():
    """Get the appropriate audio player command for the current OS"""
    system = platform.system()

    if system == "Darwin":  # macOS
        # Try ffplay first (from ffmpeg), then fallback to afplay
        if shutil.which("ffplay"):
            return "ffplay"
        if shutil.which("afplay"):
            return "afplay"
        return None
    if system == "Windows":
        # Use Windows Media Player command line
        if shutil.which("wmplayer"):
            return "wmplayer"
        if shutil.which("ffplay"):
            return "ffplay"
        return None
    if system == "Linux":
        # Try various Linux audio players
        for player in ["ffplay", "cvlc", "aplay", "paplay", "mplayer"]:
            if shutil.which(player):
                return player
        return None
    return None

def test_audio_detection():
    """Test audio player detection"""
    print("🧪 Testing Audio Player Detection")
    print("=" * 35)

    system = platform.system()
    print(f"🖥️  Operating System: {system}")

    player = get_audio_player_command()
    if player:
        print(f"✅ Audio player found: {player}")
        print(f"📂 Location: {shutil.which(player)}")

        # Test if it's executable
        try:
            import subprocess
            result = subprocess.run([player, "--help"],
                                  capture_output=True,
                                  timeout=5)
            print("🔧 Player is executable: ✅")
        except Exception as e:
            print(f"⚠️  Player test failed: {e}")

        return True
    print("❌ No compatible audio player found")
    print("💡 Install ffmpeg to get ffplay support:")
    if system == "Darwin":
        print("   brew install ffmpeg")
    elif system == "Linux":
        print("   sudo apt install ffmpeg (Ubuntu/Debian)")
        print("   sudo yum install ffmpeg (CentOS/RHEL)")
    elif system == "Windows":
        print("   Download from https://ffmpeg.org/download.html")

    return False

if __name__ == "__main__":
    success = test_audio_detection()

    if success:
        print("\n🎉 Audio playback functionality should work!")
        print("🔄 You can now restart Claude Desktop to use the play_audio_segment tool.")
    else:
        print("\n❌ Please install an audio player first.")

    exit(0 if success else 1)
