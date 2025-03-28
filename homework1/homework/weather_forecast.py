from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.min(self.data, dim=1).values, torch.max(self.data, dim=1).values)
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        raise NotImplementedError

    def find_the_largest_drop(self) -> torch.Tensor:
        return torch.min(torch.mean(self.data, dim=1)[1:] - torch.mean(self.data, dim=1)[:-1])
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        raise NotImplementedError

    ## I used chatGPT for this because I was getting stuck after finding the daily deviations
    def find_the_most_extreme_day(self) -> torch.Tensor:
        daily_means = torch.mean(self.data, dim=1)
        daily_deviations = torch.abs(self.data - daily_means.unsqueeze(1))
        max_deviation_indices = torch.argmax(daily_deviations, dim=1)
        return self.data[torch.arange(self.data.size(0)), max_deviation_indices]
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        raise NotImplementedError


    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        return torch.max(self.data[-k:], dim=1).values
        raise NotImplementedError

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        return torch.mean(self.data[-k:])
        raise NotImplementedError

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You go on a stroll next to the weather station, where this data was collected.
        You find a phone with severe water damage.
        The only thing that you can see in the screen are the
        temperature reading of one full day, right before it broke.

        You want to figure out what day it broke.

        The dataset we have starts from Monday.
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        We measure the difference using 'sum of absolute difference
        per measurement':
            d = |x1-t1| + |x2-t2| + ... + |x10-t10|

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
        differences = torch.abs(self.data - t)
        total_differences = torch.sum(differences, dim=1)
        return torch.argmin(total_differences)
        raise NotImplementedError
