import pygame
import time
import random
from sphere import Sphere
from sphere import Ship
from sphere import AmmoBox
from sphere import Mine
from random import randint
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

ship_one_initial = [100, 100]
ship_one = Ship(0, ship_one_initial[0], ship_one_initial[1], (255,0,0))

bullets = []

max_ammo_boxes = 10
max_mine_boxes = 4
ammo_boxes = []
mine_boxes = []
 
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

def game_loop():
    global pause, ship_one, ship_two
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
    #x y t 5 1
    # ship_one_initial = [100, 100, 0, 5, 1]
    
    def set_new_pose():
        ship_one_initial[0] = randint(80, display_width - 20)
        ship_one_initial[1] = randint(80, display_height - 20)

        ship_one.setValues(ship_one_initial[0], ship_one_initial[1])

    def create_new_box():
        box = None
        while box == None or ship_one.checkCollision(box):
            b_x = randint(80, display_width - 20)
            b_y = randint(80, display_height - 20)
            box = AmmoBox(b_x, b_y)
        return box

    def create_new_mine():
        box = None
        while box == None or ship_one.checkCollision(box):
            b_x = randint(80, display_width - 20)
            b_y = randint(80, display_height - 20)
            box = Mine(b_x, b_y)
        return box

    ticks = 1
    actions = 50
    for i in range(0, 10000):
        set_new_pose()

        ammo_boxes = []
        while len(ammo_boxes) < max_ammo_boxes:
            new_box = create_new_box()
            add = True
            for box in ammo_boxes:
                if box.checkCollision(new_box):
                    add = False
                    break
            if add:
                ammo_boxes.append(new_box)

        mine_boxes = []
        while len(mine_boxes) < max_mine_boxes:
            new_mine = create_new_mine()
            add = True
            for box in ammo_boxes:
                if box.checkCollision(new_mine):
                    add = False
                    break
            if add:
                for mine in mine_boxes:
                    if mine.checkCollision(new_mine):
                        add = False
                        break
            if add:        
                mine_boxes.append(new_mine)

        for j in range(0, actions * ticks):
            reward_one = 0
            reward_two = 0

            gameDisplay.fill(black)
            ship_one.render(gameDisplay)
            
            for box in ammo_boxes:
                box.render(gameDisplay)

            for box in mine_boxes:
                box.render(gameDisplay)

            # convert screen to image for tensorflow
            game_image = pygame.surfarray.array3d(gameDisplay)
            game_image = game_image.swapaxes(0,1) 
            game_image = cv2.cvtColor(game_image,cv2.COLOR_RGB2BGR)

            for box in list(ammo_boxes):
                if ship_one.checkCollision(box):
                    reward_one = 1
                    ammo_boxes.remove(box)
                    break

            for box in list(mine_boxes):
                if ship_one.checkCollision(box):
                    reward_one = -1
                    mine_boxes.remove(box)
                    break

            ship_one.tick(game_image, reward_one, (j == actions * ticks - 1) ,display_width, display_height)

            pygame.display.update()
            # clock.tick(ticks)

game_intro()
pygame.quit()
quit()
