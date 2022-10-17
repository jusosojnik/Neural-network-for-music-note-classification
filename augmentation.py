import Augmentor

p = Augmentor.Pipeline('datasets-augmented/training/Notes')
p.zoom(probability=0.5, min_factor=0.8, max_factor=1.5)
p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
p.flip_random(probability=0.3)
p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
for i in range(50):
    p.process()

p = Augmentor.Pipeline('datasets-augmented/test/Notes')
p.zoom(probability=0.5, min_factor=0.8, max_factor=1.5)
p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
p.flip_random(probability=0.3)
p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
for i in range(50):
    p.process()

p = Augmentor.Pipeline('datasets-augmented/validation/Notes')
p.zoom(probability=0.5, min_factor=0.8, max_factor=1.5)
p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
p.flip_random(probability=0.3)
p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
for i in range(50):
    p.process()



