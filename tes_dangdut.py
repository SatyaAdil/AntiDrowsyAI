import pygame
pygame.mixer.init()
pygame.mixer.music.load("static/Dangdut.mp3")
pygame.mixer.music.play()
input("Tekan Enter untuk stop...")
pygame.mixer.music.stop()
