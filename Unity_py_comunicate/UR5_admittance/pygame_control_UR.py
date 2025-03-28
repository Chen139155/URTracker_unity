# -*- encoding: utf-8 -*-
# @Author: 1391550503@qq.com
# @Date: 240221
# @Version: 1.0
# @Description: trackball

import sys
import pygame
import random
import math
import pandas as pd
# from gaze_tracking import GazeTracking
# import numpy


    
class Ball(object):
    # 定义目标小球类

    def __init__(self):
        # 定义初始化方法
        self.ball_size = 32
        self.rotation_speed = 0.1  # 默认旋转速度
        self.orbit_center = (300, 300)
        self.orbit_radius = 200
        self.ball_rect = pygame.Rect(500-self.ball_size/2, 300-self.ball_size/2, self.ball_size, self.ball_size)  # ball 矩形        # 定义目标小球的状态列表
        self.ball_status = [pygame.transform.scale(pygame.image.load('kenney_rolling-ball-assets/PNG/Default/ball_red_small.png'),(self.ball_size,self.ball_size)),
                            pygame.transform.scale(pygame.image.load('kenney_rolling-ball-assets/PNG/Default/ball_red_small_alt.png'),(self.ball_size,self.ball_size)),
                            pygame.transform.scale(pygame.image.load('kenney_rolling-ball-assets/PNG/Default/ball_blue_small.png'),(self.ball_size,self.ball_size)),
                            pygame.transform.scale(pygame.image.load('kenney_rolling-ball-assets/PNG/Default/ball_blue_small_alt.png'),(self.ball_size,self.ball_size))]

        self.status = 0  # 默认（外观）状态
        self.ball_x = 500  # 目标小球x坐标
        self.ball_y = 300  # 目标小球y坐标
        self.rect_x=self.ball_x-self.ball_size/2
        self.rect_y=self.ball_y-self.ball_size/2
        self.angle = 0

    def ball_update_auto(self):
        self.ball_x = self.orbit_center[0] + self.orbit_radius * math.cos(math.radians(self.angle))
        self.ball_y = self.orbit_center[0] + self.orbit_radius * math.sin(math.radians(self.angle))
        self.angle += self.rotation_speed
        self.rect_x=self.ball_x-self.ball_size/2
        self.rect_y=self.ball_y-self.ball_size/2
        self.ball_rect[0] = self.rect_x
        self.ball_rect[1] = self.rect_y

    def ball_update_mouse(self):
        self.ball_x, self.ball_y = pygame.mouse.get_pos()
        self.rect_x=self.ball_x-self.ball_size/2
        self.rect_y=self.ball_y-self.ball_size/2
        self.ball_rect[0] = self.rect_x
        self.ball_rect[1] = self.rect_y

def draw_background(screen):
    background_base  = pygame.image.load('kenney_rolling-ball-assets/PNG/Default/background_green.png')

    # 计算屏幕需要平铺的背景图案数量
    num_tiles_x = screen.get_width() // background_base.get_width() + 1
    num_tiles_y = screen.get_height() // background_base.get_height() + 1

    # 平铺背景图案
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            screen.blit(background_base, (i * background_base.get_width(), j * background_base.get_height()))

    # 绘制半透明圆形轨迹
    orbit_surface=screen.copy().convert()
    orbit_surface.set_alpha(100)
    pygame.draw.circle(orbit_surface, (185,184,183,10), [300, 300], 210,width=20)
    screen.blit(orbit_surface,(0,0))

# 显示开始界面
def ShowStartInterface(screen):
    screen.fill((255, 255, 255))
    draw_background(screen)
    font_surface = screen.copy().convert()
    tfont = pygame.font.Font(None, screen.get_width() // 5)
    cfont = pygame.font.Font(None, screen.get_height() // 20)
    title = tfont.render(u'TrackBall', True, ('#FF4C61'))
    content = cfont.render(u'Press any key to start', True, ('#419FDD'))
    trect = title.get_rect()
    trect.midtop = (screen.get_width() / 2, screen.get_height() / 5)
    crect = content.get_rect()
    crect.midtop = (screen.get_width() / 2, screen.get_height() / 2)
    font_surface.blit(title, trect)
    font_surface.blit(content, crect)
    screen.blit(font_surface,(0,0))
    num0 = pygame.image.load('kenney_rolling-ball-assets/PNG/Default/number_0.png')
    num1 = pygame.image.load('kenney_rolling-ball-assets/PNG/Default/number_1.png')
    num2 = pygame.image.load('kenney_rolling-ball-assets/PNG/Default/number_2.png')
    num3 = pygame.image.load('kenney_rolling-ball-assets/PNG/Default/number_3.png')
    # i=0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                # 倒计时3秒
                draw_background(screen)
                screen.blit(num3,(280,270))
                pygame.display.update()
                pygame.time.wait(1000)
                draw_background(screen)
                screen.blit(num2,(280,270))
                pygame.display.update()
                pygame.time.wait(1000)
                draw_background(screen)
                screen.blit(num1,(280,270))
                pygame.display.update()
                pygame.time.wait(1000)
                draw_background(screen)
                screen.blit(num0,(280,270))
                pygame.display.update()
                pygame.time.wait(50)
                return
        pygame.display.update()


if __name__ == '__main__':
    """主程序"""
    pygame.init()  # 初始化pygame
    pygame.font.init()  # 初始化字体
    font = pygame.font.SysFont(None, 50)  # 设置默认字体和大小

    FPS = 300
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption('Trackball')
    ShowStartInterface(screen)
    clock = pygame.time.Clock()
    # clock_for_record = pygame.time.Clock()
    timenow = 0.0
    targetball = Ball()  # 实例化
    trackball = Ball()  # 实例化
    trackball.status = 3
    score = 0
    screen.fill('white')  # 填充颜色
    mouse_flag=0 # 鼠标状态标识0表示鼠标松开，1表示鼠标按下状态
    pygame.mouse.set_visible(False)


    data_columns = ['time', 'targetball_x', 'targetball_y', 'trackball_x', 'trackball_y']
    df = pd.DataFrame(columns=data_columns)

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                df.to_excel('./output/trackballdata.xlsx')
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_flag=1
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_flag = 0
        screen.fill('white')  # 填充颜色
        draw_background(screen)
        screen.blit(targetball.ball_status[targetball.status], (targetball.rect_x, targetball.rect_y))  # 设置小球坐标
        screen.blit(trackball.ball_status[trackball.status], (trackball.rect_x, trackball.rect_y))  # 设置小球坐标
        targetball.ball_update_auto()
        if mouse_flag == 1:
            trackball.ball_update_mouse()
        timenow += clock.get_time() 
        df.loc[len(df.index)] = [timenow,targetball.ball_x,targetball.ball_y,trackball.ball_x,trackball.ball_y]
        
        pygame.display.flip()  # 更新屏幕内容
    pygame.quit()  # 退出