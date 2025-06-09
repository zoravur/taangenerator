from taangenerator.audio import *

notes_to_midi_file(generated_notes, "out.mid")
render_midi_to_wav("out.mid", "out.wav", soundfont_path)
mix_audio(taal_path, "out.wav", "mixed.wav")
play_audio("mixed.wav")
