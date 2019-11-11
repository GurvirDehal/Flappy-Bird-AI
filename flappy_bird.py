import pygame
import neat
import os
import pickle
import random
pygame.init()
pygame.font.init()

WIN_WIDTH = 550
WIN_HEIGHT  = 800
HUMAN_PLAYING = False  # Switch to true in order for a human to play
AI_TRAINING = False
GEN = -1
BIRD_IMAGES = [pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))),
               pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird2.png"))),
               pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird3.png")))]

PIPE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe.png")))
GROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "ground.png")))
BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "background.png")))

GRAVITY = 3
FONT = pygame.font.SysFont("comicsans", 50)
pygame.display.set_caption("Flappy Bird")

class Bird: 
  IMAGES = BIRD_IMAGES
  MAX_ROTATION = 25
  ROTATIONAL_VELOCITY =  10
  ANIMATION_TIME = 5

  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.tilt = 0
    self.time_count = 0
    self.velocity = 0
    self.height = self.y
    self.image_frame = 0
    self.image = self.IMAGES[0]
  
  def jump(self):
    self.velocity = -10.5
    self.time_count = 0
    self.height = self.y
  
  def move(self):
    self.time_count +=1
    displacement = self.velocity*self.time_count + 0.5*GRAVITY*self.time_count**2

    if displacement >= 16:
      displacement = 16
    
    if displacement < 0:
      displacement -= 2

    self.y = max(self.y + displacement, -75)

    if displacement < 0 or self.y < self.height + 50:
      if self.tilt < self.MAX_ROTATION:
        self.tilt = self.MAX_ROTATION
    else:
      if self.tilt > -90:
        self.tilt -= self.ROTATIONAL_VELOCITY  
  
  def draw(self, window):
    self.image_frame = (self.image_frame + 1) % (self.ANIMATION_TIME*4)
    self.image = self.IMAGES[2-abs(self.image_frame // self.ANIMATION_TIME - 2)]
    
    rotated_image = pygame.transform.rotate(self.image, self.tilt)
    new_rectangle = rotated_image.get_rect(center=self.image.get_rect(topleft = (self.x, self.y)).center)
    window.blit(rotated_image, new_rectangle.topleft)
  def get_mask(self):
    return pygame.mask.from_surface(self.image)

class Pipe:
  GAP = 200
  VELOCITY = 5

  def __init__(self, x):
    self.x = x
    self.height = 0

    self.top = 0
    self.bottom = 0
    self.PIPE_TOP = pygame.transform.flip(PIPE_IMAGE, False, True)
    self.PIPE_BOTTOM = PIPE_IMAGE

    self.passed = False
    self.set_height()

  def set_height(self):
    self.height = random.randrange(50, 450) # height from the top of the screen
    self.top = self.height - self.PIPE_TOP.get_height() # The y coordinate of the top pipe in order for its length to be self.height pixels
    self.bottom = self.height + self.GAP # The y coordinate of the bottom pipe which is the same as the bottom of the top pipe but with GAP pixels added.

  def move(self):
    self.x -= self.VELOCITY

  def draw(self, window):
    window.blit(self.PIPE_TOP, (self.x, self.top))
    window.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

  def collide(self, bird):
    bird_mask = bird.get_mask()
    top_mask = pygame.mask.from_surface(self.PIPE_TOP)
    bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
    top_offset = (self.x - bird.x, self.top - round(bird.y))
    bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

    top_collision = bird_mask.overlap(top_mask, top_offset)
    bottom_collision = bird_mask.overlap(bottom_mask, bottom_offset)

    if top_collision or bottom_collision:
      return True
    return False

class Ground:
  VELOCITY = 5
  WIDTH = GROUND_IMAGE.get_width()
  IMAGE = GROUND_IMAGE

  def __init__(self, y):
    self.y = y
    self.x1 = 0
    self.x2 = self.WIDTH

  def move(self):
    self.x1 -= self.VELOCITY
    self.x2 -= self.VELOCITY
    if self.x1 + self.WIDTH < 0:
      self.x1 = self.x2 + self.WIDTH
    elif self.x2 + self.WIDTH < 0:
      self.x2 = self.x1 + self.WIDTH

  def draw(self, window):
    window.blit(self.IMAGE, (self.x1, self.y))
    window.blit(self.IMAGE, (self.x2, self.y))


def draw_window(window, birds, pipes, ground, score, gen):
  window.blit(BACKGROUND_IMAGE, (0,0))
  for pipe in pipes:
    pipe.draw(window)

  text = FONT.render("Score: " + str(score), 1,(255,255,255))
  window.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
  if AI_TRAINING:
    gen_text = FONT.render("Gen: " + str(gen), 1,(255,255,255))
    window.blit(gen_text, (10,10))
  ground.draw(window)
  for bird in birds:
    bird.draw(window)
  pygame.display.update()


def main(genomes, config):
  global GEN
  GEN += 1
  birds = []
  ge = []
  nets = []

  for _, g in genomes:
    net = neat.nn.FeedForwardNetwork.create(g, config)
    nets.append(net)
    birds.append(Bird(230,350))
    g.fitness = 0
    ge.append(g)

  ground = Ground(730)
  pipes = [Pipe(700)]
  score = 0
  window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
  clock = pygame.time.Clock()
  run = True
  while run:
    clock.tick(30)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        run = False
        pygame.quit()
        quit()
    pipe_ind = 0
    if len(birds) > 0:
      if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
        pipe_ind = 1
    else:
      run = False
      break

    for x, bird in enumerate(birds):
      bird.move()
      ge[x].fitness += 0.1
      output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
      if output[0] > 0.5: 
        bird.jump()

    add_pipe = False
    rem = []
    for pipe in pipes:
      for x, bird in enumerate(birds):
        if pipe.collide(bird):
          ge[x].fitness -= 1
          birds.pop(x)
          nets.pop(x)
          ge.pop(x)

        if not pipe.passed and pipe.x < bird.x:
          pipe.passed = True
          add_pipe = True

      if pipe.x + pipe.PIPE_TOP.get_width() < 0:
        rem.append(pipe) # queue pipe to be removed once it is on screen
      
      pipe.move()

    if add_pipe:
      score += 1
      for g in ge:
        g.fitness += 5 # adds 5 fitness to birds who've made it through the pipes
      pipes.append(Pipe(700))

    for r in rem:
      pipes.remove(r) 
    for x, bird in enumerate(birds):
      if bird.y + bird.image.get_height() >= 730 or bird.y < 0:
        birds.pop(x)
        nets.pop(x)
        ge.pop(x)

    ground.move()
    draw_window(window, birds, pipes, ground, score, GEN)
    if AI_TRAINING and score > 300:
      break


def human_mode():
  bird = [Bird(230,350)]
  ground = Ground(730)
  pipes = [Pipe(700)]
  score = 0
  window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
  clock = pygame.time.Clock()
  run = True
  while run:
    clock.tick(30)
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        run = False
        pygame.quit()
        quit()
      if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        bird[0].jump()

    bird[0].move()
    add_pipe = False
    rem = []
    for pipe in pipes:
      if pipe.collide(bird[0]):
        bird.pop(0)
        run = False
        break

      if not pipe.passed and pipe.x < bird[0].x:
        pipe.passed = True
        add_pipe = True

      if pipe.x + pipe.PIPE_TOP.get_width() < 0:
        rem.append(pipe) # queue pipe to be removed once it is on screen
      
      pipe.move()
    if not run:
      break
    if add_pipe:
      score += 1
      pipes.append(Pipe(700))

    for r in rem:
      pipes.remove(r) 

    if bird[0].y + bird[0].image.get_height() >= 730:
      bird.pop(0)
      run = False
      break
    ground.move()
    draw_window(window, bird, pipes, ground, score, GEN)


def run(config):
  population = neat.Population(config)
  population.add_reporter(neat.StdOutReporter(True))
  statistics = neat.StatisticsReporter()
  population.add_reporter(statistics)
  winner = population.run(main ,10)
   # Save the winner.
  with open('best.pickle', 'wb') as f:
      pickle.dump(winner, f)


if HUMAN_PLAYING:
  while True:
    human_mode()
elif __name__ == "__main__":
  local_dir = os.path.dirname(__file__)
  config_path = os.path.join(local_dir, "config-feedforward.txt")
  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
  if AI_TRAINING:
    run(config)
  else:
    # load the winner
    with open('best.pickle', 'rb') as f:
        c = pickle.load(f)
    main([(0, c)], config)