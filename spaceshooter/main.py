import pygame
import time
import random
from sphere import Sphere
from sphere import Ship
import cv2
 
pygame.init()
 
display_width = 800
display_height = 800
 
black = (0,0,0)
white = (255,255,255)

red = (200,0,0)
green = (0,200,0)

bright_red = (255,0,0)
bright_green = (0,255,0)
 
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Spaceshooter')
clock = pygame.time.Clock()

pause = False

ship_one_initial = [100, 100, 0, 5, 1]
ship_one = Ship(ship_one_initial[0], ship_one_initial[1], ship_one_initial[2], ship_one_initial[3], ship_one_initial[4])

ship_two_initial = [400, 400, 225, 5, 1]
ship_two = Ship(ship_two_initial[0], ship_two_initial[1], ship_two_initial[2], ship_two_initial[3], ship_two_initial[4])
bullets = []
 
def text_objects(text, font):
    textSurface = font.render(text, True, black)
    return textSurface, textSurface.get_rect() 

def button(msg,x,y,w,h,ic,ac,action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(gameDisplay, ac,(x,y,w,h))
        if click[0] == 1 and action != None:
            action()         
    else:
        pygame.draw.rect(gameDisplay, ic,(x,y,w,h))
    smallText = pygame.font.SysFont("comicsansms",20)
    textSurf, textRect = text_objects(msg, smallText)
    textRect.center = ( (x+(w/2)), (y+(h/2)) )
    gameDisplay.blit(textSurf, textRect)
    

def quitgame():
    pygame.quit()
    quit()

def unpause():
    global pause
    pygame.mixer.music.unpause()
    pause = False
    

def paused():
    ############
    pygame.mixer.music.pause()
    #############
    largeText = pygame.font.SysFont("comicsansms",115)
    TextSurf, TextRect = text_objects("Paused", largeText)
    TextRect.center = ((display_width/2),(display_height/2))
    gameDisplay.blit(TextSurf, TextRect)
    

    while pause:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()


        button("Continue",150,450,100,50,green,bright_green,unpause)
        button("Quit",550,450,100,50,red,bright_red,quitgame)

        pygame.display.update()
        clock.tick(15)   


def game_intro():

    intro = True

    while intro:
        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
          
        game_loop()      

def get_actions(actions):
    global pause
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                actions['left'] = True
            if event.key == pygame.K_RIGHT:
                actions['right'] = True
            if event.key == pygame.K_UP:
                actions['up'] = True
            if event.key == pygame.K_DOWN:
                actions['down'] = True
            if event.key == pygame.K_SPACE:
                bullet = ship_one.fire()
                bullets.append(bullet)
            if event.key == pygame.K_p:
                pause = True
                paused()
                
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                actions['left'] = False
            if event.key == pygame.K_RIGHT:
                actions['right'] = False
            if event.key == pygame.K_UP:
                actions['up'] = False
            if event.key == pygame.K_DOWN:
                actions['down'] = False

    if actions['up']:
        ship_one.accelerate()   
    if actions['down']:
        ship_one.deccelerate()
    if actions['left']:
        ship_one.rotateLeft()
    if actions['right']:
        ship_one.rotateRight()

def game_loop():
    global pause, ship_one, ship_two, bullets
    ############
    # pygame.mixer.music.load('jazz.wav')
    # pygame.mixer.music.play(-1)
    ############
    gameExit = False
    
    actions = {}
    actions['up'] = False
    actions['down'] = False
    actions['left'] = False
    actions['right'] = False

    ship_one.setValues(ship_one_initial[0], ship_one_initial[1], ship_one_initial[2], ship_one_initial[3], ship_one_initial[4])
    ship_two.setValues(ship_two_initial[0], ship_two_initial[1], ship_two_initial[2], ship_two_initial[3], ship_two_initial[4])
    bullets = []

    reward_one = 0
    reward_two = 0
    while not gameExit:
        gameDisplay.fill(black)
        ship_one.render(gameDisplay)
        ship_two.render(gameDisplay)
        for bullet in bullets:
            bullet.render(gameDisplay)

        # convert screen to image for tensorflow
        game_image = pygame.surfarray.array3d(gameDisplay)
        game_image = game_image.swapaxes(0,1) 
        game_image = cv2.cvtColor(game_image,cv2.COLOR_RGB2BGR)
        # cv2.imshow("Test", imgdata)
        # cv2.waitKey(1)

        if ship_one.checkCollision(ship_two):
            print('Both loose')
            reward_one = -100
            reward_two = -100
            gameExit = True
        else:
            for bullet in list(bullets):
                if bullet.tick(display_width, display_height):
                    bullets.remove(bullet)
                    break

                if ship_one.checkCollision(bullet):
                    reward_two = 100
                    reward_one = -100
                    print('Ship two wins!')
                    gameExit = True
                    break
                if ship_two.checkCollision(bullet):
                    reward_one = 100
                    reward_two = -100
                    print('Ship one wins!')
                    gameExit = True
                    break

        #TODO Thread this 
        bullet_one = ship_one.tick(display_width, display_height, game_image, reward_one)
        if bullet_one != None:
            bullets.append(bullet_one)

        bullet_two = ship_two.tick(display_width, display_height, game_image, reward_two)
        if bullet_two != None:
            bullets.append(bullet_two)

        pygame.display.update()
        clock.tick(10)

game_intro()
pygame.quit()
quit()
