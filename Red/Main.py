import tensorflow
import matplotlib.pyplot as plt
from auxiliar import *
import warnings
import os
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


allfolders = os.listdir()
dirfolders = []
for fols in allfolders:
	if '_A' in fols or '_B' in fols or '_C' in fols or '_D' in fols or '_E' in fols or '_F' in fols or '_G' in fols or '_H' in fols:
		dirfolders.append(fols)

print(dirfolders)

data = Dataset()
test = Dataset()

set_pvi = []
for p in dirfolders:
	set_pose = read_file('./'+p+'/pose.out')[0:-1]
	
	final_pose = [float(round(item*2)/2) for item in set_pose[-1]]
	set_velocity = read_file('./'+p+'/velocity.out')[0:-1]
	set_images = get_paths('./'+p+'/Images')[0:-1]

	cont = 0
	for pose, vel, img in zip(set_pose, set_velocity, set_images):
		step_img = unify_images(set_images, cont)

		set_pvi.append(Step(pose, vel, step_img, final_pose))
		cont += 1

np.random.shuffle(set_pvi)
imgs, px, py, pz, angs, angs2 = data.load_data(set_pvi)

NN = Neural_Network(5, 0.001, 32, 5, False)

NN.training([imgs, py, pz, angs, angs2], data.oh_vel)
NN.save()

x_table = np.zeros([5, 5], dtype=int)
y_table = np.zeros([5, 5], dtype=int)
z_table = np.zeros([5, 5], dtype=int)
gz_table = np.zeros([5, 5], dtype=int)

for i in range(len(set_pvi)):
	m = NN.predict(imgs[i], py[i], pz[i], angs[i], angs2[i])


	x = np.argmax(m[0])
	x_table[data.aux[i][0]][x] += 1
	
	y = np.argmax(m[1])
	y_table[data.aux[i][1]][y] += 1

	z = np.argmax(m[2])
	z_table[data.aux[i][2]][z] += 1

	gz = np.argmax(m[3])
	gz_table[data.aux[i][3]][gz] += 1
	
	if i % 100 == 0:
		print('Numero {}'.format(i))

xtab = resize_matrix(x_table, 40)
ytab = resize_matrix(y_table, 40)
ztab = resize_matrix(z_table, 40)
gztab = resize_matrix(gz_table, 40)

aux = np.isnan(xtab)
xtab[aux] = 0.0
aux = np.isnan(ytab)
ytab[aux] = 0.0
aux = np.isnan(ztab)
ztab[aux] = 0.0
aux = np.isnan(gztab)
gztab[aux] = 0.0


plt.imshow(xtab, cmap='gray')
plt.show()
plt.imshow(ytab, cmap='gray')
plt.show()
plt.imshow(ztab, cmap='gray')
plt.show()
plt.imshow(gztab, cmap='gray')
plt.show()

# Guardar resultados como imagen
'''
cv2.imwrite('xtab.png', xtab*255)
cv2.imwrite('ytab.png', ytab*255)
cv2.imwrite('ztab.png', ztab*255)
cv2.imwrite('gztab.png', gztab*255)
'''