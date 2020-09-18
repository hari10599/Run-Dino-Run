from selenium import webdriver
from selenium.webdriver.common.keys import Keys
class GameDriver:
    def __init__(self):
        self.webDriver = webdriver.Chrome(executable_path = '/Users/harikrishna/Documents/Projects/Run Dino Run/chromedriver')
        self.webDriver.get('chrome://dino')
        time.sleep(1)
        self.webDriver.execute_script("Runner.config.ACCELERATION=0")
        self.webDriver.execute_script("document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'")

    def jump(self):
        self.webDriver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)   

    def restartGame(self):
        self.webDriver.execute_script("Runner.instance_.restart()")

    def isRunning(self):
        return self.webDriver.execute_script("return Runner.instance_.crashed")
