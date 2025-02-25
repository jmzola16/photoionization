# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:27:13 2024

@author: James Zola
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib
import bisect

def front_curvature_paper(R, z_start, length, title):
    #z_edge = np.linspace(z_start, z_start + length, 1001)
    z_edge = np.linspace(0, length, 1001)
    r_edge = np.linspace(-1.5*R, 1.5*R, 201)
    z = np.diff(z_edge)/2 + z_edge[0:-1]
    r = np.diff(r_edge)/2 + r_edge[0:-1]

    ave_pt_len = np.zeros((200, 1000))
    tol = 0.01

    x = [[], [], [], [], []]
    y = [[], [], [], [], []]
    val = [[], [], [], [], []]

    for i in range(len(z)):
        for j in range((len(r))):
            def f(w, beta):
                return w*np.sqrt(w**2 + r[j]**2 - 2*w*r[j]*np.cos(beta) + z[i]**2)/R
            ave_pt_len[j, i] = 1/(np.pi*R**2)*integrate.dblquad(f, 0, 2*np.pi, 0, R)[0]
        
            for p in range(1, 6):
                if abs(ave_pt_len[j, i] - p) < tol:
                    #x[p - 1].append((z[i] - z_start)/R)
                    x[p - 1].append(z[i]/R)
                    y[p - 1].append(r[j]/R)
                    val[p - 1].append(ave_pt_len[j, i])
            

    fig, ax =  plt.subplots()
    #im = ax.pcolormesh((z_edge - z_start)/R, r_edge/R, ave_pt_len)
    im = ax.pcolormesh(z_edge/R, r_edge/R, ave_pt_len)
    for p in range(5):
        plt.scatter(x[p], y[p], s=1, c="r")
    ax.axis('equal')
    plt.xlabel("Distance (Source Radii)")
    plt.ylabel("Radius (Source Radius)")
    plt.title("Front curvature for " + title)
    fig.colorbar(im, ax=ax)
    plt.show()
    
def front_curvature_reemission(R, alpha, length, title):
    #z_edge = np.linspace(z_start, z_start + length, 1001)
    z_edge = np.linspace(0, length, 1001)
    r_edge = np.linspace(-1.5*R, 1.5*R, 201)
    z = np.diff(z_edge)/2 + z_edge[0:-1]
    r = np.diff(r_edge)/2 + r_edge[0:-1]

    ave_pt_len = np.zeros((200, 1000))
    tol = 0.01

    x = [[], [], [], [], []]
    y = [[], [], [], [], []]
    val = [[], [], [], [], []]

    for i in range(len(z)):
        for j in range((len(r))):
            def f(w, beta):
                return w*np.sqrt(w**2 + r[j]**2 - 2*w*r[j]*np.cos(beta) + z[i]**2)/R
            ave_pt_len[j, i] = 1/(np.pi*R**2)*integrate.dblquad(f, 0, 2*np.pi, 0, R)[0]
            
            def f_cyl(x, phi):
                #ind = bisect.bisect_left(z_edge, x) - 1
                return np.sqrt((z[i] - x)**2 + r[j]**2 + R**2 - 2*r[j]*R*np.cos(phi)) #+ ave_pt_len[-1, ind]
            
            ave_pt_len[j, i] += alpha/(np.pi*R*z[i])*integrate.dblquad(f_cyl, 0, 2*np.pi, 0, z[i])[0]
            
            for p in range(1, 6):
                if abs(ave_pt_len[j, i] - p) < tol:
                    #x[p - 1].append((z[i] - z_start)/R)
                    x[p - 1].append(z[i]/R)
                    y[p - 1].append(r[j]/R)
                    val[p - 1].append(ave_pt_len[j, i])
            

    fig, ax =  plt.subplots()
    #im = ax.pcolormesh((z_edge - z_start)/R, r_edge/R, ave_pt_len)
    im = ax.pcolormesh(z_edge/R, r_edge/R, ave_pt_len)
    for p in range(5):
        plt.scatter(x[p], y[p], s=1, c="r")
    ax.axis('equal')
    plt.xlabel("Distance (Source Radii)")
    plt.ylabel("Radius (Source Radius)")
    plt.title("Front curvature for " + title)
    fig.colorbar(im, ax=ax)
    plt.show()
    
def front_curvature_paper_data(R, length):
    R = 0.1
    length = 0.7
    z_edge = np.linspace(0, length, 1001)
    r_edge = np.linspace(0, 1.5*R, 101)
    z = np.diff(z_edge)/2 + z_edge[0:-1]
    r = np.diff(r_edge)/2 + r_edge[0:-1]

    ave_pt_len = np.zeros((200, 1000))
    tol = 0.01

    x = [[], [], [], [], []]
    y = [[], [], [], [], []]
    val = [[], [], [], [], []]

    for i in range(len(z)):
        for j in range((len(r))):
            def f(w, beta):
                return w*np.sqrt(w**2 + r[j]**2 - 2*w*r[j]*np.cos(beta) + z[i]**2)/R
            ave_pt_len[j, i] = 1/(np.pi*R**2)*integrate.dblquad(f, 0, 2*np.pi, 0, R)[0]
        
            for p in range(1, 6):
                if abs(ave_pt_len[j, i] - p) < tol:
                    #x[p - 1].append((z[i] - z_start)/R)
                    x[p - 1].append(z[i]/R)
                    y[p - 1].append(r[j]/R)
                    val[p - 1].append(ave_pt_len[j, i])
                    
    return ave_pt_len, z_edge, r_edge, x, y, val   