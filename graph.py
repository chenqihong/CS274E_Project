import matplotlib.pyplot as plt

f = open("log.txt", "r")

train_loss = []
valid_loss = []

train_kld = []
valid_kld = []

lstm_loss = []

for line in f.readlines():
    if "lstmloss" in line:
        lstm_loss.append(float(line[37:]))
    if "trainloss" in line:
        train_loss.append(float(line[37:]))
    if "validloss" in line:
        valid_loss.append(float(line[37:]))
    if "kld" in line:
        if "train" in line:
            train_kld.append(float(line[line.find("kld") + 4:]))
        if "valid" in line:
            valid_kld.append(float(line[line.find("kld") + 4:]))


it = range(0, 35000, 50)

plt.plot(it, train_loss, label="Train")
plt.plot(it, valid_loss, label="Valid")


plt.xlabel('Iteration')
plt.ylabel('Loss')


plt.title('Iteration vs Loss')

plt.legend()
plt.show()

lstm_loss=[x/25 for x in lstm_loss]

plt.plot(it, train_kld, label="Train")
plt.plot(it, valid_kld, label="Valid")
plt.plot(it, lstm_loss, label="LSTM")

plt.xlabel('Iteration')
plt.ylabel('KLD')

plt.title('Iteration vs KLD vs LSTM')

plt.legend()
plt.show()
