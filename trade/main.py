
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height


def main():
    rectangle = Rectangle(10, 20)

    print(f'The value of width is {rectangle.width}')
    print(f'The value of height is {rectangle.height}')


if __name__ == '__main__':
    main()