#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GLFW/glfw3.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <queue>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>
#include <chrono>  // Asigură-te că ai inclus această bibliotecă

const int WINDOW_WIDTH = 1000;
const int WINDOW_HEIGHT = 800;
const int GRID_SIZE = 10;
float OBSTACLE_COVERAGE = 0.25f;
bool robotMoving = false;
std::chrono::steady_clock::time_point startTime;
auto programStart = std::chrono::steady_clock::now(); // Salvează timpul de start al programului

auto programEnd = std::chrono::steady_clock::now(); // Timpul de final
std::chrono::duration<double> totalExecutionTime = programEnd - programStart;
struct GameObject {
    float x, y, width, height;
};

std::vector<GameObject> obstacles;
std::vector<std::pair<float, float>> path;
GameObject robot, goal;
std::vector<GameObject> waypoints;

float randomFloat(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

bool checkCollision(const GameObject& a, const GameObject& b) {
    return (a.x < b.x + b.width &&
            a.x + a.width > b.x &&
            a.y < b.y + b.height &&
            a.y + a.height > b.y);
}

bool isValidPosition(const GameObject& obj, const std::vector<GameObject>& others) {
    for (const auto& existing : others) {
        if (checkCollision(obj, existing)) {
            return false;
        }
    }
    return true;
}

float heuristic(float x1, float y1, float x2, float y2) {
    return fabs(x1 - x2) + fabs(y1 - y2);
}

std::vector<std::pair<float, float>> findPath(float startX, float startY, float goalX, float goalY) {
    struct Node {
        float x, y, g, h;
        Node* parent;
        bool operator>(const Node& other) const {
            return (g + h) > (other.g + other.h);
        }
    };
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openSet;
    std::vector<std::vector<bool>> visited(WINDOW_WIDTH / GRID_SIZE,
        std::vector<bool>(WINDOW_HEIGHT / GRID_SIZE, false));
    openSet.push({startX, startY, 0.0f, heuristic(startX, startY, goalX, goalY), nullptr});
    std::vector<std::pair<int,int>> directions = {{0,1},{1,0},{0,-1},{-1,0}};
    while (!openSet.empty()) {
        Node current = openSet.top();
        openSet.pop();
        int gridX = (int)(current.x / GRID_SIZE);
        int gridY = (int)(current.y / GRID_SIZE);
        if (gridX < 0 || gridX >= (WINDOW_WIDTH / GRID_SIZE) ||
            gridY < 0 || gridY >= (WINDOW_HEIGHT / GRID_SIZE)) {
            continue;
        }
        if (visited[gridX][gridY]) {
            continue;
        }
        visited[gridX][gridY] = true;
        if (fabs(current.x - goalX) < GRID_SIZE && fabs(current.y - goalY) < GRID_SIZE) {
            std::vector<std::pair<float, float>> resultPath;
            while (current.parent) {
                resultPath.push_back({current.x, current.y});
                current = *current.parent;
            }
            std::reverse(resultPath.begin(), resultPath.end());
            return resultPath;
        }
        for (auto [dx, dy] : directions) {
            float nx = current.x + dx * GRID_SIZE;
            float ny = current.y + dy * GRID_SIZE;
            GameObject temp = {nx, ny, robot.width, robot.height};
            if (!isValidPosition(temp, obstacles) || nx < 0 || ny < 0 ||
                nx >= WINDOW_WIDTH || ny >= WINDOW_HEIGHT) {
                continue;
            }
            float gNew = current.g + GRID_SIZE;
            float hNew = heuristic(nx, ny, goalX, goalY);
            openSet.push({nx, ny, gNew, hNew, new Node(current)});
        }
    }
    return {};
}

float getPathLength(const std::vector<std::pair<float,float>>& p) {
    if (p.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    float length = 0.0f;
    for (size_t i = 1; i < p.size(); i++) {
        float dx = p[i].first - p[i-1].first;
        float dy = p[i].second - p[i-1].second;
        length += std::sqrt(dx*dx + dy*dy);
    }
    return length;
}

static std::vector<std::vector<float>> costMatrix;

void buildDistanceMatrix(const std::vector<std::pair<float,float>>& coords) {
    int totalNodes = (int)coords.size();
    costMatrix.assign(totalNodes, std::vector<float>(totalNodes, 0.0f));
    for (int i = 0; i < totalNodes; i++) {
        for (int j = 0; j < totalNodes; j++) {
            if (i == j) {
                costMatrix[i][j] = 0.0f;
                continue;
            }
            auto p = findPath(coords[i].first, coords[i].second,
                              coords[j].first, coords[j].second);
            costMatrix[i][j] = getPathLength(p);
        }
    }
}

std::vector<int> solveTSP(int N) {
    int totalNodes = N+2;
    int FULL_MASK = (1 << N) - 1;
    std::vector<std::vector<float>> dp(1 << N, std::vector<float>(N+1, std::numeric_limits<float>::infinity()));
    std::vector<std::vector<int>> parent(1 << N, std::vector<int>(N+1, -1));
    for (int i = 1; i <= N; i++) {
        float c = costMatrix[0][i];
        if (c < std::numeric_limits<float>::infinity()) {
            dp[1 << (i-1)][i] = c;
            parent[1 << (i-1)][i] = 0;
        }
    }
    for (int mask = 0; mask < (1 << N); mask++) {
        for (int i = 1; i <= N; i++) {
            if (dp[mask][i] == std::numeric_limits<float>::infinity()) continue;
            if (((mask >> (i-1)) & 1) == 0) continue;
            for (int j = 1; j <= N; j++) {
                if ((mask >> (j-1)) & 1) {
                    continue;
                }
                float c = costMatrix[i][j];
                if (c == std::numeric_limits<float>::infinity()) continue;
                int newMask = mask | (1 << (j-1));
                float newCost = dp[mask][i] + c;
                if (newCost < dp[newMask][j]) {
                    dp[newMask][j] = newCost;
                    parent[newMask][j] = i;
                }
            }
        }
    }
    float bestCost = std::numeric_limits<float>::infinity();
    int bestLast = -1;
    for (int i = 1; i <= N; i++) {
        float c = dp[FULL_MASK][i];
        float cToGoal = costMatrix[i][N+1];
        if (c < std::numeric_limits<float>::infinity() && cToGoal < std::numeric_limits<float>::infinity()) {
            float total = c + cToGoal;
            if (total < bestCost) {
                bestCost = total;
                bestLast = i;
            }
        }
    }
    std::vector<int> order;
    if (bestLast == -1) {
        order.push_back(0);
        order.push_back(N+1);
        return order;
    }
    int curNode = bestLast;
    int curMask = FULL_MASK;
    while (curNode != 0) {
        order.push_back(curNode);
        int p = parent[curMask][curNode];
        curMask = curMask & ~(1 << (curNode-1));
        curNode = p;
    }
    order.push_back(0);
    std::reverse(order.begin(), order.end());
    order.push_back(N+1);
    return order;
}

std::vector<std::pair<float, float>> buildFinalPath(const std::vector<int>& bestOrder,
                                                    const std::vector<std::pair<float,float>>& coords) {
    std::vector<std::pair<float, float>> full;
    for (size_t k = 0; k+1 < bestOrder.size(); k++) {
        int u = bestOrder[k];
        int v = bestOrder[k+1];
        auto segment = findPath(coords[u].first, coords[u].second,
                                coords[v].first, coords[v].second);
        if (!segment.empty()) {
            if (k == 0) {
                full.insert(full.end(), segment.begin(), segment.end());
            } else {
                full.insert(full.end(), segment.begin()+1, segment.end());
            }
        }
    }
    return full;
}

void generateGameObjects(int waypointCount) {
    srand((unsigned)time(0));
    float totalMapArea = WINDOW_WIDTH * WINDOW_HEIGHT;
    float targetObstacleArea = totalMapArea * OBSTACLE_COVERAGE;
    float currentObstacleArea = 0.0f;
    while (currentObstacleArea < targetObstacleArea) {
        GameObject obstacle;
        do {
            obstacle = {
                randomFloat(0, WINDOW_WIDTH - 100),
                randomFloat(0, WINDOW_HEIGHT - 100),
                100,
                100
            };
        } while (!isValidPosition(obstacle, obstacles));
        obstacles.push_back(obstacle);
        currentObstacleArea += (obstacle.width * obstacle.height);
    }
    do {
        robot = {
            (float)(WINDOW_WIDTH - GRID_SIZE),
            0.0f,
            15.0f,
            15.0f
        };
    } while (!isValidPosition(robot, obstacles));
    do {
        goal = {
            0.0f,
            (float)(WINDOW_HEIGHT - GRID_SIZE),
            10.0f,
            10.0f
        };
    } while (!isValidPosition(goal, obstacles));
    for (int i = 0; i < waypointCount; i++) {
        GameObject wp;
        bool valid = false;
        while (!valid) {
            wp = {
                randomFloat(0, WINDOW_WIDTH - 15),
                randomFloat(0, WINDOW_HEIGHT - 15),
                15.0f,
                15.0f
            };
            std::vector<GameObject> checkObjects = obstacles;
            checkObjects.push_back(robot);
            checkObjects.push_back(goal);
            checkObjects.insert(checkObjects.end(), waypoints.begin(), waypoints.end());
            if (isValidPosition(wp, checkObjects)) {
                valid = true;
            }
        }
        waypoints.push_back(wp);
    }
}
std::chrono::steady_clock::time_point  endTime;
bool timerStarted = false;  // Flag to ensure timer starts only once

void startRobotMovementOptimal() {
    robotMoving = true;
    timerStarted = false; // Ensure the timer starts correctly
    std::vector<std::pair<float,float>> coords;
    coords.push_back({robot.x, robot.y});
    for (auto &w : waypoints) {
        coords.push_back({w.x, w.y});
    }
    coords.push_back({goal.x, goal.y});
    int N = (int)waypoints.size();
    buildDistanceMatrix(coords);
    std::vector<int> bestOrder = solveTSP(N);
    std::vector<std::pair<float,float>> fullPath = buildFinalPath(bestOrder, coords);
    path = fullPath;
}

void updateRobot() {
    if (!path.empty()) {
        if (!timerStarted) {
            startTime = std::chrono::steady_clock::now(); // Start timp doar pentru robot
            timerStarted = true;
        }

        auto [nx, ny] = path.front();
        float dx = nx - robot.x;
        float dy = ny - robot.y;
        float dist = sqrt(dx * dx + dy * dy);
        float speed = 0.3f;

        if (dist < speed) {
            robot.x = nx;
            robot.y = ny;
            path.erase(path.begin());
        } else {
            robot.x += (dx / dist) * speed;
            robot.y += (dy / dist) * speed;
        }
    }

    // **Șterge punctele (waypoints) atinse**
    GameObject robotBox = {robot.x, robot.y, robot.width, robot.height};
    for (int i = (int)waypoints.size() - 1; i >= 0; i--) {
        if (checkCollision(robotBox, waypoints[i])) {
            waypoints.erase(waypoints.begin() + i);
        }
    }

    // **Verificare finalizare traseu**
    float epsilon = 11.0f; // Prag de detecție a obiectivului final
    float goalDist = sqrt(pow(robot.x - goal.x, 2) + pow(robot.y - goal.y, 2));

    if (path.empty() && goalDist < epsilon && timerStarted) {
        endTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedTime = endTime - startTime;

        // **Calculează timpul total de execuție**
        auto programEnd = std::chrono::steady_clock::now();
        std::chrono::duration<double> totalExecutionTime = programEnd - programStart;

        // **Afișare rezultate**
        printf("Robot reached the goal in %.4f seconds.\n", elapsedTime.count());
        printf("Total execution time: %.4f seconds.\n", totalExecutionTime.count());

        timerStarted = false;

        // **Închide programul după ce robotul ajunge la final**
        exit(0);
    }
}





void drawRectangle(float x, float y, float w, float h, float r, float g, float b) {
    glColor3f(r, g, b);
    glBegin(GL_QUADS);
    glVertex2f(x,     y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x,     y + h);
    glEnd();
}

void renderGame() {
    glClear(GL_COLOR_BUFFER_BIT);
    for (auto &obs : obstacles) {
        drawRectangle(obs.x, obs.y, obs.width, obs.height, 0.5f, 0.5f, 0.5f);
    }
    for (auto &wp : waypoints) {
        drawRectangle(wp.x, wp.y, wp.width, wp.height, 0.0f, 1.0f, 0.0f);
    }
    drawRectangle(robot.x, robot.y, robot.width, robot.height, 0.0f, 0.0f, 1.0f);
    drawRectangle(goal.x, goal.y, goal.width, goal.height, 1.0f, 0.0f, 0.0f);
    glFlush();
}

void runGame(int waypointCount) {
    if (!glfwInit()) return;
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "TSP + A* Robot", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return;
    }
    glfwMakeContextCurrent(window);
    glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1);
    generateGameObjects(waypointCount);
    startRobotMovementOptimal();
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);
        updateRobot();
        renderGame();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
}

int main() {
    programStart = std::chrono::steady_clock::now(); // Salvează timpul de start al programului

    int N;
    std::cout << "Number of waypoints: ";
    std::cin >> N;
    runGame(N);

    return 0;
}

