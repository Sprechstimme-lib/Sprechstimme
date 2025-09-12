import sprechstimme as sp

sp.new("lead")
sp.create("lead", wavetype=sp.waves.sawtooth)

track = sp.Track(bpm=100)
track.add("lead", notes=[60], duration=1)
track.add("lead", notes=[64], duration=1)
track.add("lead", notes=[67], duration=2)

track.play()
track.export("my_song.wav")