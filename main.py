from login import main as login_main
from UIV2 import main as ui_main

def main():
    logged_in = login_main()
    if logged_in:
        ui_main()

if __name__ == '__main__': main()