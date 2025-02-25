from typing import Literal

import torch
from torch.utils.data import Dataset
import random


class ShapePairDataset(Dataset):
    def __init__(
        self,
        num_samples,
        image_size=32,
        max_radius=4,
        max_square_size=8,
        seed=42,
        mode: Literal["train", "val", "test"] = "train",
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.radius = max_radius
        self.square_size = max_square_size
        self.mode = mode

        random.seed(seed)
        torch.manual_seed(seed)

        self.Y, self.X = torch.meshgrid(
            torch.arange(image_size), torch.arange(image_size), indexing="ij"
        )

        self.items = [self._generate_data() for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def _generate_data(self):
        img1, shape1 = self._generate_image()
        img2, shape2 = self._generate_image()

        return img1, shape1, img2, shape2

    def __getitem__(self, idx):
        img1, shape1, img2, shape2 = self.items[idx]
        label = 1.0 if shape1 == shape2 else 0.0
        if self.mode == "test":
            return img1, int(shape1 == "circle"), img2, int(shape2 == "circle")
        return img1, img2, torch.tensor(label)

    def _generate_image(self):
        is_circle = random.random() > 0.5
        color = self._random_color()
        background_color = torch.clamp(self._random_color() + color, 0.0, 1.0)

        if is_circle:
            return self._draw_circle(color, background_color), "circle"
        else:
            return self._draw_square(color, background_color), "square"

    def _random_color(self):
        return torch.FloatTensor(3).uniform_(0, 1)

    def _draw_circle(self, color, background_color):
        x = random.randint(self.radius, self.image_size - self.radius - 1)
        y = random.randint(self.radius, self.image_size - self.radius - 1)

        mask = (self.X - x) ** 2 + (self.Y - y) ** 2 <= self.radius**2
        image = torch.zeros(3, self.image_size, self.image_size)

        image[:, :] = background_color.view(3, 1, 1)
        image[:, mask] = color.view(3, 1)
        return image

    def _draw_square(self, color, background_color):
        square_size = random.randint(2, self.square_size)
        start_x = random.randint(0, self.image_size - square_size)
        start_y = random.randint(0, self.image_size - square_size)

        image = torch.zeros(3, self.image_size, self.image_size)

        image[:, :] = background_color.view(3, 1, 1)
        image[:, start_x : start_x + square_size, start_y : start_y + square_size] = (
            color.view(3, 1, 1)
        )
        return image


class ColorClassificationDataset(Dataset):
    def __init__(self, num_samples, image_size=32):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.radius = 4
        self.square_size = 8

        self.Y, self.X = torch.meshgrid(
            torch.arange(image_size), torch.arange(image_size), indexing="ij"
        )
        self.items = [self._generate_data() for i in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, label = self.items[idx]
        return img, torch.tensor(label)

    def _random_color(self):
        return torch.FloatTensor(3).uniform_(0, 1)

    def _generate_data(self):
        img1, shape = self._generate_image()
        if shape == "circle":
            label = 0
        else:
            label = 1
        return img1, label

    def _generate_image(self):
        is_circle = random.random() > 0.5

        if is_circle:
            return self._draw_circle(), "circle"
        else:
            return self._draw_square(), "square"

    def _draw_circle(self):
        color = self._random_color()
        background_color = self._random_color()
        x = random.randint(self.radius, self.image_size - self.radius - 1)
        y = random.randint(self.radius, self.image_size - self.radius - 1)

        mask = (self.X - x) ** 2 + (self.Y - y) ** 2 <= self.radius**2
        image = torch.zeros(3, self.image_size, self.image_size)
        image[:, :] = background_color.view(3, 1, 1)
        image[:, mask] = color.view(3, 1)

        return image

    def _draw_square(self):
        color = self._random_color()
        background_color = self._random_color()
        square_size = random.randint(1, self.square_size)
        start_x = random.randint(0, self.image_size - self.square_size)
        start_y = random.randint(0, self.image_size - self.square_size)

        image = torch.zeros(3, self.image_size, self.image_size)

        image[:, :] = background_color.view(3, 1, 1)
        image[:, start_x : start_x + square_size, start_y : start_y + square_size] = (
            color.view(3, 1, 1)
        )
        return image
