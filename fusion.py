def fuse_emotions(face_emotion, face_conf, voice_emotion, voice_conf):
    # Voice dominates for anger
    if voice_emotion == "Angry" and voice_conf >= 0.5:
        return "Angry"

    # If both agree
    if face_emotion == voice_emotion:
        return face_emotion

    # Otherwise weighted decision
    if voice_conf > face_conf:
        return voice_emotion
    else:
        return face_emotion
