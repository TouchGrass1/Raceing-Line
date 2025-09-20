class Button:
    def __init__(self, x, y, w, h, label):
        self.rect = pg.Rect(x, y, w, h)
        self.checked = False
        self.label = label
    def draw(self, screen, font):
        pg.draw.rect(screen, (88, 101, 242), self.rect, width = 2)
        if self.checked:
            pg.draw.rect(screen, (255, 255, 255), self.rect)
        label_text = font.render(self.label, True, (0,0,0))
        label_rect = label_text.get_rect(center=self.rect.center)
        screen.blit(label_text, (self.rect.x + 3, self.rect.y + 3))