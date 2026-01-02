import cv2
import pyttsx3
from spellchecker import SpellChecker
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# === Model Setup ===
spell = SpellChecker()
engine = pyttsx3.init()

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trocr_model.to(device)

# === Spell Correction ===
def spell_correct(text):
    words = text.split()
    corrected_words = []
    for word in words:
        corrected = spell.correction(word)
        if corrected:
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

# === OCR from Image using TrOCR ===
def extract_text_from_image(image_bytes):
    try:
        image = Image.open(image_bytes).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        generated_ids = trocr_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text.strip()
    except Exception as e:
        print("[ERROR] TrOCR extraction failed:", e)
        return ""

# === Speak Text ===
def speak(text):
    engine.say(text)
    engine.runAndWait()

# === Main Function ===
def main():
    print("[INFO] Starting webcam stream... Press 'q' to quit.")
    
    cap = cv2.VideoCapture(0)  # Use local webcam

    if not cap.isOpened():
        print("[ERROR] Failed to open webcam.")
        return

    last_text = ""
    motion_detected = False
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        cv2.imshow("Live Feed", frame)

        # === Motion detection ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        diff = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_level = np.sum(thresh) / 255

        if motion_level > 5000:
            motion_detected = True
        else:
            motion_detected = False

        prev_frame = gray

        # === Process when motion is detected ===
        if motion_detected:
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                try:
                    from io import BytesIO
                    image_stream = BytesIO(buffer.tobytes())

                    extracted = extract_text_from_image(image_stream)

                    if extracted and extracted != last_text:
                        print("🔍 Extracted:", extracted)

                        corrected = spell_correct(extracted)
                        print("✅ Corrected:", corrected)

                        speak(corrected)

                        last_text = extracted
                except Exception as e:
                    print("[ERROR] OCR processing failed:", e)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
