import os
import sys
import sqlite3
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# get xy positions from focalplane database
connection = sqlite3.connect('focalplane_pb2a.db')
cursor = connection.cursor()
data = cursor.execute("SELECT bolo_name, x_pos, y_pos from focalplane").fetchall()
bolo_names = [d[0] for d in data]
x_poss = np.array([d[1] for d in data])
y_poss = np.array([d[2] for d in data])
connection.close()

# for now, cherry pick a single data point from Fred's ray tracing to use for a naive conversion
radius = 180.8 # mm
offset = -2.26 # degrees
conv = offset / radius
pointing_offsets_simple_x = x_poss * conv
pointing_offsets_simple_y = y_poss * conv

plt.figure(figsize=(10,10))
q = plt.quiver(x_poss, y_poss, pointing_offsets_simple_x, pointing_offsets_simple_y,
               units='inches', scale_units='inches', scale=20.0, pivot='tail')
plt.quiverkey(q, X=0.8, Y=0.9, U=2, label='2 degrees', labelpos='E')
plt.axes().set_aspect('equal')
plt.xlabel('x position on the focal plane, viewed from the sky side [mm]')
plt.ylabel('y position on the focal plane, viewed from the sky side [mm]')
plt.title('PB2a pointing offsets from naive linear estimate')
plt.show()

all_sql_statements = []
all_sql_statements.append("BEGIN TRANSACTION;") # is this necessary?
all_sql_statements.append("PRAGMA foreign_keys=OFF;") # no idea what this does


connection = sqlite3.connect('offsets_pb2a.db')
cursor = connection.cursor()

create_statement = """CREATE TABLE IF NOT EXISTS naive (
bolo_name text PRIMARY_KEY,
delta_x real NOT NULL,
delta_y real NOT NULL
);"""
all_sql_statements.append(create_statement)

for bolo_name, offset_x, offset_y in zip(bolo_names, pointing_offsets_simple_x, pointing_offsets_simple_y):
    insert_statement = "INSERT INTO naive(bolo_name, delta_x, delta_y) VALUES ('{:s}', {:f}, {:f});".format(bolo_name, offset_x, offset_y)
    all_sql_statements.append(insert_statement)

all_sql_statements.append('COMMIT;') # is this necessary?
msg = '\n'.join(all_sql_statements)
print(msg)

for cmd in all_sql_statements:
    cursor.execute(cmd)



