import pygame, copy

screen = pygame.display.set_mode((720, 720)) 
    
pygame.display.set_caption('Neural Network') 
screen.fill((255, 255, 255)) 
clock = pygame.time.Clock()
pygame.display.flip() 
  
running = True
hitboxes = [[0] * 32 for _ in range(32)]
inputs = copy.deepcopy(hitboxes)

TILE_SIZE = 15
TILE_X, TILE_Y = 100, 100
for i in range(32):
    for j in range(32):
        rect = pygame.Rect(TILE_X + i*TILE_SIZE, TILE_Y + j*TILE_SIZE, TILE_SIZE, TILE_SIZE)
        hitboxes[i][j] = rect

while running: 
    for event in pygame.event.get():     
        if event.type == pygame.QUIT: 
            running = False
    
    click, _, _ = pygame.mouse.get_pressed()
    mpos = pygame.mouse.get_pos()
    for i in range(32):
        for j in range(32):
            if hitboxes[i][j].collidepoint(mpos) and click: 
                inputs[i][j] = 1
            pygame.draw.rect(screen, (0,0,0), hitboxes[i][j], inputs[i][j] ^ 1)

    pygame.display.flip() 
    clock.tick(24)
