import numpy as np

N = 10
M = 4
noise = 0.2
max_iter = 100

def show_image(x):
    for i in range(N):
        for j in range(N):
            if x[i*N+j] == 1:
                print("#", end="")
            else:
                print(".", end="")
        print()
    print()

def random_image():
    return np.random.choice([-1, 1], size=N*N)

def noisy_image(x):
    return np.where(np.random.random(size=N*N) < noise, -x, x)

def potential(x, w, i):
    return np.dot(w[i], x)

def update(x, w, i):
    return np.sign(potential(x, w, i))

def energy(x, w):
    return -0.5 * np.dot(x, np.dot(w, x))

images = [random_image() for _ in range(M)]

print("Imágenes originales:")
for x in images:
    show_image(x)

noisy_images = [noisy_image(x) for x in images]

print("Imágenes con ruido:")
for x in noisy_images:
    show_image(x)

w = np.zeros((N*N, N*N))
for x in images:
    w += np.outer(x, x)
w /= N*N
np.fill_diagonal(w, 0)

# Vector que representa la imagen del aro "C" y su centro "A"
x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
              -1, 1, 1, 1, 1, 1, 1, 1, -1, 1,
              -1, 1, -1, -1, -1, -1, -1, -1, -1, 1,
              -1, 1, -1, 1, 1, 1, 1, 1, -1, 1,
              -1, 1, -1, 1, 1, 1, 1, 1, -1, 1,
              -1, 1, -1, 1, 1, 1, 1, 1, -1, 1,
              -1, 1, -1, -1, -1, -1, -1, -1, -1, 1,
              -1, 1, 1, 1, 1, 1, 1, 1, -1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

w += np.outer(x, x)
w /= N*N
np.fill_diagonal(w, 0)

x = np.pad(x, (0, N*N - len(x)))
x = np.reshape(x, (N, N))
x_noisy = noisy_image(x.flatten())
x_noisy = np.reshape(x_noisy, (N*N,))
x = np.reshape(x, (N*N,))

x_old = x_noisy.copy()

iter = 0

while True:
    i = np.random.randint(N*N)
    x_old[i] = update(x_old, w, i)
    iter += 1
    e = energy(x_old, w)
    print(f"Iteración {iter}:")
    show_image(x_old)
    print(f"Energía: {e}")
    if np.array_equal(x_old, x) or iter >= max_iter:
        break

def centroid(x):
    sum_x = 0
    sum_y = 0
    sum_v = 0
    for i in range(N):
        for j in range(N):
            v = x[i*N+j]
            if v == 1:
                sum_x += j * v
                sum_y += i * v
                sum_v += v
    x_c = sum_x / sum_v
    y_c = sum_y / sum_v
    return x_c, y_c

x_c, y_c = centroid(x_old)
print(f"Coordenadas X e Y del centro: ({x_c}, {y_c})")
