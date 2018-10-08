import pandas as pd
import matplotlib.pyplot as plt

run_id = 4

log_file = '~/Desktop/train_{run_id}.log'.format(run_id=run_id)
logs = pd.read_json(log_file, lines=True)

plt.figure(figsize=(20, 40))

plt.plot(logs.step[logs.loss.notnull()], logs.loss[logs.loss.notnull()], label="training loss")

plt.plot(logs.step[logs.valid_loss.notnull()], logs.valid_loss[logs.valid_loss.notnull()], label="validation loss")
plt.plot(logs.step[logs.valid_loss.notnull()], logs.jaccard[logs.valid_loss.notnull()]*100, label="IoU")

plt.xlabel("step")
plt.legend(loc='lower right')
# plt.tight_layout()
plt.show()
