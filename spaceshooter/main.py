import pygame
import time
import random
from sphere import Sphere
from sphere import Ship
import cv2
 
pygame.init()

#############
# crash_sound = pygame.mixer.Sound("crash.wav")
#############
 
display_width = 800
display_height = 600
 
black = (0,0,0)
white = (255,255,255)

red = (200,0,0)
green = (0,200,0)

bright_red = (255,0,0)
bright_green = (0,255,0)
 
block_color = (53,115,255)
 
car_width = 73
 
gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('A bit Racey')
clock = pygame.time.Clock()
 
carImg = pygame.image.load('racecar.png')
gameIcon = pygame.image.load('racecar.png')

pygame.display.set_icon(gameIcon)

pause = False
#crash = True

# sphere = Sphere(100, 100, 0, 10, 0.5)
sphere = Ship(100, 100, 0, 10, 0.5)
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
                
        gameDisplay.fill(white)
        largeText = pygame.font.SysFont("comicsansms",115)
        TextSurf, TextRect = text_objects("A bit Racey", largeText)
        TextRect.center = ((display_width/2),(display_height/2))
        gameDisplay.blit(TextSurf, TextRect)

        button("GO!",150,450,100,50,green,bright_green,game_loop)
        button("Quit",550,450,100,50,red,bright_red,quitgame)

        pygame.display.update()
        clock.tick(15)
    
def game_loop():
    global pause
    ############
    # pygame.mixer.music.load('jazz.wav')
    # pygame.mixer.music.play(-1)
    ############
    gameExit = False

    key_pressed = {}
    key_pressed['up'] = False
    key_pressed['down'] = False
    key_pressed['left'] = False
    key_pressed['right'] = False
 
    while not gameExit:
 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    key_pressed['left'] = True
                if event.key == pygame.K_RIGHT:
                    key_pressed['right'] = True
                if event.key == pygame.K_UP:
                    key_pressed['up'] = True
                if event.key == pygame.K_DOWN:
                    key_pressed['down'] = True
                if event.key == pygame.K_SPACE:
                    bullet = sphere.fire()
                    bullets.append(bullet)
                if event.key == pygame.K_p:
                    pause = True
                    paused()
                    
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    key_pressed['left'] = False
                if event.key == pygame.K_RIGHT:
                    key_pressed['right'] = False
                if event.key == pygame.K_UP:
                    key_pressed['up'] = False
                if event.key == pygame.K_DOWN:
                    key_pressed['down'] = False
 
        if key_pressed['up']:
            sphere.accelerate()
        if key_pressed['down']:
            sphere.deccelerate()
        if key_pressed['left']:
            sphere.rotateLeft()
        if key_pressed['right']:
            sphere.rotateRight()

        gameDisplay.fill(white)
        sphere.render(gameDisplay)
        for bullet in bullets:
            bullet.render(gameDisplay)

        # convert screen to image for tensorflow
        imgdata = pygame.surfarray.array3d(gameDisplay)
        imgdata = imgdata.swapaxes(0,1) 
        imgdata = cv2.cvtColor(imgdata,cv2.COLOR_RGB2BGR)
        cv2.imshow("Test", imgdata)
        cv2.waitKey(1)

        for bullet in bullets:
            bullet.tick()

        sphere.tick()
        
        
        pygame.display.update()
        clock.tick(60)

game_intro()
game_loop()
pygame.quit()
quit()