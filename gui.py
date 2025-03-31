import pygame, copy, nn_model
print("Let's make this program not dumb")
# nn_model.train_model()
pygame.init()

screen = pygame.display.set_mode((720, 720))     
pygame.display.set_caption('Smart python program!') 
screen.fill((255, 255, 255)) 
clock = pygame.time.Clock()
pygame.display.flip() 
  
running = True
hitboxes = [[0] * 28 for _ in range(28)]
inputs = copy.deepcopy(hitboxes)
font = pygame.font.Font(pygame.font.get_default_font(), 40)

TILE_SIZE = 15
TILE_X, TILE_Y = 100, 100
for i in range(28):
    for j in range(28):
        rect = pygame.Rect(TILE_X + i*TILE_SIZE, TILE_Y + j*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        hitboxes[i][j] = rect

def generate_button(text, rect):
    txt = font.render(text, True, (0,0,0))
    hitbox = txt.get_rect()
    hitbox.center = rect.center
    pygame.draw.rect(screen, (255,255,255), rect)
    screen.blit(txt, hitbox)

def generate_heading(text, center):
    title = font.render(text, True, (0,0,0))
    titlehitbox = title.get_rect()
    titlehitbox.center = center
    screen.blit(title, titlehitbox)


while running: 
    for event in pygame.event.get():     
        if event.type == pygame.QUIT: 
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r: 
                inputs = [[0] * 28 for _ in range(28)]
                screen.fill((255,255,255))
    
    click, _, _ = pygame.mouse.get_pressed()
    mpos = pygame.mouse.get_pos()
    for i in range(28):
        for j in range(28):
            if hitboxes[i][j].collidepoint(mpos) and click: 
                inputs[i][j] = 1
            pygame.draw.rect(screen, (0,0,0), hitboxes[i][j], inputs[i][j] ^ 1)

    guess_b = pygame.Rect(360, 600, 50, 50)
    generate_button("Guess the number", guess_b)

    if guess_b.collidepoint(mpos) and click:
        generate_heading(f"I think you wrote a {nn_model.test_on_handwriting(inputs)}", (360, 50))

    pygame.display.flip() 
    clock.tick(120)

