import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)


def visualise_graph(pos, mass, projection='3d', **kwargs):
    """Visualise a graph in 2D or 3D.

    Parameters
    ----------
    pos : pd.DataFrame
        A DataFrame with columns 'x', 'y', and 'z' containing the positions of the nodes.
    mass : pd.Series
        A Series containing the mass of the nodes.
    projection : str, optional
        The projection to use, either '2d' or '3d'. Default is '3d'.
    **kwargs
        Additional keyword arguments to pass to the plotting function.
    """
    kwargs = {'color': 'black', 'fontsize': 12, 'alpha': 0.6, 'cmap': 'plasma', 'vmin': 9.5, 'vmax': 13}

    # update the kwargs with the provided kwargs
    kwargs.update(kwargs)

    if projection == '2d':
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.scatter(pos[:, 0], pos[:, 1], s=0.01, c=kwargs['color'], alpha=kwargs['alpha'])
        ax.set_xlabel('X [Mpc]', size=kwargs['fontsize'])
        ax.set_ylabel('Y [Mpc]', size=kwargs['fontsize'])

        return fig, ax
    elif projection == '3d':
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=300, subplot_kw={'projection': '3d'})

        sc = ax[1].scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=3.5**(mass - 10), c=mass, alpha=kwargs['alpha'], cmap=kwargs['cmap'], edgecolor='k', linewidths=0.1, vmin=kwargs['vmin'], vmax=kwargs['vmax'])

        ax[1].set_xlabel('X [Mpc]', size=kwargs['fontsize'])
        ax[1].set_ylabel('Y [Mpc]', size=kwargs['fontsize'])
        ax[1].set_zlabel('Z [Mpc]', size=kwargs['fontsize'])
        ax[1].xaxis.set_tick_params(labelsize=12)
        ax[1].yaxis.set_tick_params(labelsize=12)
        ax[1].zaxis.set_tick_params(labelsize=12)
        ax[1].xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax[1].yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax[1].zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax[1].xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        ax[1].yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        ax[1].zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)

        # Define the region of interest (ROI) for the zoom-in
        roi_center = np.array([502, 598, 400])
        # roi_center = np.array([817.406250,	920.983887,	569.646301	])
        roi_size = 10
        roi_mask = (
            (pos[:, 0] > roi_center[0] - roi_size) & (pos[:, 0] < roi_center[0] + roi_size) &
            (pos[:, 1] > roi_center[1] - roi_size) & (pos[:, 1] < roi_center[1] + roi_size) &
            (pos[:, 2] > roi_center[2] - roi_size) & (pos[:, 2] < roi_center[2] + roi_size)
        )
        pos_roi = pos[roi_mask]
        mass_roi = mass[roi_mask]

        # Add the zoom-in plot
        zoom_ax = ax[0]
        zoom_sc = zoom_ax.scatter(pos_roi[:, 0], pos_roi[:, 1], pos_roi[:, 2], c=mass_roi, s=6**(mass_roi - 10), cmap='plasma', vmin=9.5, vmax=13, edgecolor='k', linewidths=0.1, alpha=0.6)


        # Axis labels and adjustments for the zoom-in plot
        zoom_ax.set_xlim(roi_center[0] - roi_size, roi_center[0] + roi_size)
        zoom_ax.set_ylim(roi_center[1] - roi_size, roi_center[1] + roi_size)
        zoom_ax.set_zlim(roi_center[2] - roi_size, roi_center[2] + roi_size)

        # zoom_ax.set_xlabel('X [Mpc]', fontsize=16)
        # zoom_ax.set_ylabel('Y [Mpc]', fontsize=16)
        # zoom_ax.set_zlabel('Z [Mpc]', fontsize=16)
        zoom_ax.xaxis.set_tick_params(labelsize=12)
        zoom_ax.yaxis.set_tick_params(labelsize=12)
        zoom_ax.zaxis.set_tick_params(labelsize=12)
        zoom_ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        zoom_ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        zoom_ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        zoom_ax.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        zoom_ax.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        zoom_ax.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)


        # Add a box to highlight the ROI in the main plot
        ax[1].plot([roi_center[0] - roi_size, roi_center[0] + roi_size], [roi_center[1] - roi_size, roi_center[1] - roi_size], [roi_center[2] - roi_size, roi_center[2] - roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] - roi_size, roi_center[0] + roi_size], [roi_center[1] + roi_size, roi_center[1] + roi_size], [roi_center[2] - roi_size, roi_center[2] - roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] - roi_size, roi_center[0] + roi_size], [roi_center[1] - roi_size, roi_center[1] - roi_size], [roi_center[2] + roi_size, roi_center[2] + roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] - roi_size, roi_center[0] + roi_size], [roi_center[1] + roi_size, roi_center[1] + roi_size], [roi_center[2] + roi_size, roi_center[2] + roi_size], color='r', zorder=10, linewidth=0.5)

        ax[1].plot([roi_center[0] - roi_size, roi_center[0] - roi_size], [roi_center[1] - roi_size, roi_center[1] + roi_size], [roi_center[2] - roi_size, roi_center[2] - roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] + roi_size, roi_center[0] + roi_size], [roi_center[1] - roi_size, roi_center[1] + roi_size], [roi_center[2] - roi_size, roi_center[2] - roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] - roi_size, roi_center[0] - roi_size], [roi_center[1] - roi_size, roi_center[1] + roi_size], [roi_center[2] + roi_size, roi_center[2] + roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] + roi_size, roi_center[0] + roi_size], [roi_center[1] - roi_size, roi_center[1] + roi_size], [roi_center[2] + roi_size, roi_center[2] + roi_size], color='r', zorder=10, linewidth=0.5)

        ax[1].plot([roi_center[0] - roi_size, roi_center[0] - roi_size], [roi_center[1] - roi_size, roi_center[1] - roi_size], [roi_center[2] - roi_size, roi_center[2] + roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] + roi_size, roi_center[0] + roi_size], [roi_center[1] - roi_size, roi_center[1] - roi_size], [roi_center[2] - roi_size, roi_center[2] + roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] - roi_size, roi_center[0] - roi_size], [roi_center[1] + roi_size, roi_center[1] + roi_size], [roi_center[2] - roi_size, roi_center[2] + roi_size], color='r', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center[0] + roi_size, roi_center[0] + roi_size], [roi_center[1] + roi_size, roi_center[1] + roi_size], [roi_center[2] - roi_size, roi_center[2] + roi_size], color='r', zorder=10, linewidth=0.5)


        # Define the region of interest (ROI) for the second zoom-in
        roi_center2 = np.array([258.772461,	391.547852,	666.322998]) # ID 0
        roi_center2 = np.array([669.756836,	148.352570,	155.136078])
        roi_size2 = 20
        roi_mask2 = (
            (pos[:, 0] > roi_center2[0] - roi_size2) & (pos[:, 0] < roi_center2[0] + roi_size2) &
            (pos[:, 1] > roi_center2[1] - roi_size2) & (pos[:, 1] < roi_center2[1] + roi_size2) &
            (pos[:, 2] > roi_center2[2] - roi_size2) & (pos[:, 2] < roi_center2[2] + roi_size2)
        )
        pos_roi2 = pos[roi_mask2]
        mass_roi2 = mass[roi_mask2]

        # Add the second zoom-in plot
        zoom_ax2 = ax[2]
        zoom_sc2 = zoom_ax2.scatter(pos_roi2[:, 0], pos_roi2[:, 1], pos_roi2[:, 2], c=mass_roi2, s=6**(mass_roi2 - 10), cmap='plasma', vmin=9.5, vmax=13, edgecolor='k', linewidths=0.1, alpha=0.6)
        zoom_ax2.view_init(elev=30, azim=40)  # Adjust orientation here

        # Axis labels and adjustments for the second zoom-in plot
        zoom_ax2.set_xlim(roi_center2[0] - roi_size2, roi_center2[0] + roi_size2)
        zoom_ax2.set_ylim(roi_center2[1] - roi_size2, roi_center2[1] + roi_size2)
        zoom_ax2.set_zlim(roi_center2[2] - roi_size2, roi_center2[2] + roi_size2)

        # zoom_ax2.set_xlabel('X [Mpc]', fontsize=16)
        # zoom_ax2.set_ylabel('Y [Mpc]', fontsize=16)
        # zoom_ax2.set_zlabel('Z [Mpc]', fontsize=16)
        zoom_ax2.xaxis.set_tick_params(labelsize=12)
        zoom_ax2.yaxis.set_tick_params(labelsize=12)
        zoom_ax2.zaxis.set_tick_params(labelsize=12)
        zoom_ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        zoom_ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        zoom_ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        zoom_ax2.xaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        zoom_ax2.yaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)
        zoom_ax2.zaxis._axinfo["grid"]['color'] =  (0.5,0.5,0.5,0.2)

        # Add a box to highlight the ROI in the main plot for the second zoom-in
        ax[1].plot([roi_center2[0] - roi_size2, roi_center2[0] + roi_size2], [roi_center2[1] - roi_size2, roi_center2[1] - roi_size2], [roi_center2[2] - roi_size2, roi_center2[2] - roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] - roi_size2, roi_center2[0] + roi_size2], [roi_center2[1] + roi_size2, roi_center2[1] + roi_size2], [roi_center2[2] - roi_size2, roi_center2[2] - roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] - roi_size2, roi_center2[0] + roi_size2], [roi_center2[1] - roi_size2, roi_center2[1] - roi_size2], [roi_center2[2] + roi_size2, roi_center2[2] + roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] - roi_size2, roi_center2[0] + roi_size2], [roi_center2[1] + roi_size2, roi_center2[1] + roi_size2], [roi_center2[2] + roi_size2, roi_center2[2] + roi_size2], color='b', zorder=10, linewidth=0.5)

        ax[1].plot([roi_center2[0] - roi_size2, roi_center2[0] - roi_size2], [roi_center2[1] - roi_size2, roi_center2[1] + roi_size2], [roi_center2[2] - roi_size2, roi_center2[2] - roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] + roi_size2, roi_center2[0] + roi_size2], [roi_center2[1] - roi_size2, roi_center2[1] + roi_size2], [roi_center2[2] - roi_size2, roi_center2[2] - roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] - roi_size2, roi_center2[0] - roi_size2], [roi_center2[1] - roi_size2, roi_center2[1] + roi_size2], [roi_center2[2] + roi_size2, roi_center2[2] + roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] + roi_size2, roi_center2[0] + roi_size2], [roi_center2[1] - roi_size2, roi_center2[1] + roi_size2], [roi_center2[2] + roi_size2, roi_center2[2] + roi_size2], color='b', zorder=10, linewidth=0.5)

        ax[1].plot([roi_center2[0] - roi_size2, roi_center2[0] - roi_size2], [roi_center2[1] - roi_size2, roi_center2[1] - roi_size2], [roi_center2[2] - roi_size2, roi_center2[2] + roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] + roi_size2, roi_center2[0] + roi_size2], [roi_center2[1] - roi_size2, roi_center2[1] - roi_size2], [roi_center2[2] - roi_size2, roi_center2[2] + roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] - roi_size2, roi_center2[0] - roi_size2], [roi_center2[1] + roi_size2, roi_center2[1] + roi_size2], [roi_center2[2] - roi_size2, roi_center2[2] + roi_size2], color='b', zorder=10, linewidth=0.5)
        ax[1].plot([roi_center2[0] + roi_size2, roi_center2[0] + roi_size2], [roi_center2[1] + roi_size2, roi_center2[1] + roi_size2], [roi_center2[2] - roi_size2, roi_center2[2] + roi_size2], color='b', zorder=10, linewidth=0.5)


        # Create a common colorbar on top of the figure
        cbar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.03])  # Position: [left, bottom, width, height]
        cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('log($M_{\\rm *} / M_{\\odot})$', fontsize=16)
        cbar.ax.tick_params(labelsize=12)

        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        return fig, ax


    else:
        raise ValueError(f'Invalid projection: {projection}')


def is_central_plot(fig, ax, halomass, stellarmass, is_central, colorbar=True):
    """Add a plot to visualise the central and satellite galaxies.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to add the plot to.
    ax : matplotlib.axes.Axes
        The axes to add the plot to.
    halomass : pd.Series
        A Series containing the mass of the halos.
    stellarmass : pd.Series
        A Series containing the mass of the galaxies.
    is_central : pd.Series
        A Series containing the boolean flag for central galaxies.
    """
    ax.scatter(halomass, stellarmass, s=0.1, c=is_central, cmap='coolwarm', alpha=0.8)
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo} / M_{\odot})$')
    ax.set_ylabel(r'$\log_{10}(M_{\rm star} / M_{\odot})$')

    if colorbar:
        # Add a colorbar
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Central galaxy', rotation=270, labelpad=15)
        cbar.set_ticks([0, 1])

    return fig, ax


def plot_true_vs_pred(y_true, y_pred, c, ax, fig, title=None, cmap='viridis', clabel=None, colorbar=True):
    sc = ax.scatter(y_true, y_pred, c=c, cmap=cmap, s=1, vmin=0, vmax=10)
    ax.plot([10.5, 15], [10.5, 15], 'k--')
    ax.set_xlabel(r"True log ($M_{\rm{halo}}/M_\odot$)", fontsize=16)
    ax.set_ylabel(r"Predicted log ($M_{\rm{halo}}/M_\odot$)", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(10.5, 15)
    ax.set_ylim(10.5, 15)

    # tick size
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

    # set aspect ratio to 1
    ax.set_aspect('equal', 'box')

    # add colorbar with some extra space on the right
    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, pad=0.06)
        cbar.set_label(clabel, fontsize=16)
    
    # add grid lines
    ax.grid(True, alpha=0.15)

    # add text of RMSE dex
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    # ax.text(0.05, 0.9, f'RMSE={rmse:.3f} dex', fontsize=14, transform=ax.transAxes)

    # add a scatter region
    # _, edges = np.histogram(y_true, bins=15)
    # scatter, scatter_lower = [], []
    # for i in range(len(edges) - 1):
    #     mask = (y_true >= edges[i]) & (y_true < edges[i + 1])
    #     scatter.append(np.std(y_true[mask]) + edges[i])
    #     scatter_lower.append(edges[i] - np.std(y_true[mask]))
    
    # ax.fill_between(edges[:-1], scatter_lower, scatter, color='gray', alpha=0.3, zorder=10)
    
    return ax