#include "lib/script.h"
#include <queue>
#include <vector>

#pragma once


const std::vector<std::vector<std::vector<float>>> s_locationBounds = {
    {
        { 245.f,		277.f,		239.f,	154.f,	80.f,	-173.f ,	-94.f,	7.f,	-18.f },
{ -995.f,	-889.f,		-834.f,	-845.f,	-985.f,	-855.f ,	-732.f,	-766.f,	-912.f },
    },	// special case for 6 minute demo
    {
        { 190.5051f,	-263.2976f,	-324.604f,	336.7958f,	454.3408f },	// x-coords
{ -580.1099f,	-627.6421f,	-1065.248f,	-1068.885f,	-751.1761f },	// y-coords
{ 133.7935f, -970.5508f },	// point in polygon - used for flood-fill algorithm
    },	// downtown
        /*{
        { 314.841f,		-132.75f,	-212.515f,	232.525f },		// x-coords
        { -835.796f,	-685.729f,	-903.577f,	-1068.683f },	// y-coords
        { 133.7935f, -970.5508f },	// point in polygon - used for flood-fill algorithm
        },	// downtown*/
    {
        { -294.127f,	277.186f,	358.727f,	202.9238f, },
{ -1448.299f,	-1887.974f,	-1513.72f,	-1278.168f },
{ 110.797f, -1409.013f },
//scentral
    },
};

// loc:				the id of area to focus on, see settings.h => s_locationBounds
// num_points:		number of points to generate
// dist:			minimum allowed distance between points
// init:			

static bool in_bounds(float x, float y, int loc, std::vector<std::vector<char>> &poly_grid)
{
    int min_x = (int)round(*std::min_element(std::begin(s_locationBounds[loc][0]), std::end(s_locationBounds[loc][0])));
    int max_x = (int)round(*std::max_element(std::begin(s_locationBounds[loc][0]), std::end(s_locationBounds[loc][0])));
    int min_y = (int)round(*std::min_element(std::begin(s_locationBounds[loc][1]), std::end(s_locationBounds[loc][1])));
    int max_y = (int)round(*std::max_element(std::begin(s_locationBounds[loc][1]), std::end(s_locationBounds[loc][1])));

    if (poly_grid.empty())
    {
        std::vector<char> temp_1;
        std::pair<short, short> temp_2;
        std::queue<std::pair<short, short>> fill_queue;

        for (int i = min_y; i <= max_y; i++)
            temp_1.push_back(0);
        for (int i = min_x; i <= max_x; i++)
            poly_grid.push_back(temp_1);
        temp_1.clear();

        for (int point = 0; point < s_locationBounds[loc][0].size(); point++)
        {
            int next_point = (point + 1) % s_locationBounds[loc][0].size();
            float diff_x = s_locationBounds[loc][0][next_point] - s_locationBounds[loc][0][point];
            float diff_y = s_locationBounds[loc][1][next_point] - s_locationBounds[loc][1][point];

            if (abs(diff_x) >= abs(diff_y))
            {
                float slope = diff_y / diff_x;
                float intercept = s_locationBounds[loc][1][point] - slope * s_locationBounds[loc][0][point];

                for (float x = std::min(s_locationBounds[loc][0][point], s_locationBounds[loc][0][next_point]);
                    x <= std::max(s_locationBounds[loc][0][point], s_locationBounds[loc][0][next_point]); x++)
                {
                    int x_index = int(round(x)) - min_x;
                    int y_index = int(round(slope * x + intercept)) - min_y;
                    poly_grid[x_index][y_index] = 2;
                }
            }
            else
            {
                float slope = diff_x / diff_y;
                float intercept = s_locationBounds[loc][0][point] - slope * s_locationBounds[loc][1][point];

                for (float y = std::min(s_locationBounds[loc][1][point], s_locationBounds[loc][1][next_point]);
                    y <= std::max(s_locationBounds[loc][1][point], s_locationBounds[loc][1][next_point]); y++)
                {
                    int y_index = int(round(y)) - min_y;
                    int x_index = int(round(slope * y + intercept)) - min_x;
                    poly_grid[x_index][y_index] = 2;
                }
            }
        }

        fill_queue.push({ int(round(s_locationBounds[loc][2][0])) - min_x, int(round(s_locationBounds[loc][2][1])) - min_y });
        while (fill_queue.size() > 0)
        {
            temp_2 = fill_queue.front();
            fill_queue.pop();

            if (poly_grid[temp_2.first][temp_2.second] > 0)
                continue;

            poly_grid[temp_2.first][temp_2.second] = 1;

            if (temp_2.first > 0)
                fill_queue.push({ temp_2.first - 1, temp_2.second });
            if (temp_2.first < poly_grid.size() - 1)
                fill_queue.push({ temp_2.first + 1, temp_2.second });
            if (temp_2.second > 0)
                fill_queue.push({ temp_2.first, temp_2.second - 1 });
            if (temp_2.second < poly_grid[0].size() - 1)
                fill_queue.push({ temp_2.first, temp_2.second + 1 });
        }

        std::string temp_3 = "";
        for (int i = int(poly_grid.size()) - 1; i > 0; i--)
        {
            temp_3 = "";
            for (int j = int(poly_grid[i].size()) - 1; j > 0; j--)
                temp_3 += poly_grid[i][j] > 0 ? "#" : ".";
        }
    }

    int check_x = int(round(x - min_x));
    int check_y = int(round(y - min_y));
    return check_x > 0 && check_y > 0 &&
        check_x < poly_grid.size() - 1 &&
        check_y < poly_grid[0].size() - 1 &&
        poly_grid[check_x][check_y] > 0;
}

static std::vector<std::pair<float, float>> generate_n_random_points(int loc, std::vector<std::vector<char>> &poly_grid, int num_points, float min_dist = 100, std::vector<std::pair<float, float>> init = {})
{
    std::vector<std::pair<float, float>> points;

    int min_x = (int)round(*std::min_element(std::begin(s_locationBounds[loc][0]), std::end(s_locationBounds[loc][0])));
    int max_x = (int)round(*std::max_element(std::begin(s_locationBounds[loc][0]), std::end(s_locationBounds[loc][0])));
    int min_y = (int)round(*std::min_element(std::begin(s_locationBounds[loc][1]), std::end(s_locationBounds[loc][1])));
    int max_y = (int)round(*std::max_element(std::begin(s_locationBounds[loc][1]), std::end(s_locationBounds[loc][1])));

    if (!init.empty())
        points.insert(points.end(), init.begin(), init.end());
    for (int i = 0; i < num_points; i++)
    {
        bool valid = true;
        std::pair<float, float> temp_3;

        do {
            temp_3.first = float(min_x + rand() % int(max_x - min_x));
            temp_3.second = float(min_y + rand() % int(max_y - min_y));
        } while (!in_bounds(temp_3.first, temp_3.second, loc, poly_grid));

        for (int j = 0; j < i; j++)
            if (pow(points[j].first - temp_3.first, 2) + pow(points[j].second - temp_3.second, 2) < pow(min_dist, 2))
            {
                valid = false;
                break;
            }

        if (valid)
            points.push_back(temp_3);
        else
            i--;
    }
    if (!init.empty())
        for (int i = 0; i < init.size(); i++)
            points.erase(points.begin());

    return points;
}