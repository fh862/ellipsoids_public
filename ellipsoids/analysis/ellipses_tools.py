#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:14:17 2024

@author: fangfang
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import math
from skimage.measure import EllipseModel
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

#%%
def angles_to_2Dchromatic_directions(theta_deg, normalize=True):
    """
    Convert planar angles (theta) to 2D chromatic direction vectors.

    Convention
    ----------
    theta : direction angle in the x-y plane, degrees in [0, 360)

    Unit vector (before optional renormalization)
    --------------------------------------------
      x = cos(theta)
      y = sin(theta)

    Parameters
    ----------
    theta_deg : (N,) array-like
        Direction angles in degrees.
    normalize : bool, default True
        If True, renormalize each direction to unit length (mainly a safety step).

    Returns
    -------
    chromatic_directions : (N, 2) ndarray
        2D direction vectors.
    """
    theta_deg = np.asarray(theta_deg, dtype=float).ravel()
    theta_rad = np.deg2rad(theta_deg)

    chromatic_directions = np.column_stack([
        np.cos(theta_rad),
        np.sin(theta_rad),
    ])  # (N, 2)

    if normalize:
        n = np.linalg.norm(chromatic_directions, axis=1)
        valid = n > 0
        if not np.all(valid):
            n_bad = np.sum(~valid)
            print(f"[angles_to_chromatic_directions_2D] Warning: {n_bad} direction(s) "
                  "have zero norm; leaving them as zero vectors.")
        chromatic_directions[valid] /= n[valid, None]

    return chromatic_directions

def ellParams_to_covMat(radii, evecs):
    """
    Convert ellipsoid parameters into a covariance matrix. This function uses
    the radii and orientation (eigenvectors) of an ellipsoid to construct its
    covariance matrix.
    
    Args:
    radii (numpy.ndarray): An array containing the radii of the ellipsoid along 
        its principal axes. These radii can be interpreted as the standard 
        deviations of the distribution along each principal axis of the ellipsoid.
    evecs (numpy.ndarray): A 2D array representing the eigenvectors of the ellipsoid.
        These eigenvectors define the orientation of the ellipsoid and form the 
        columns of this matrix. The eigenvectors should be orthogonal and 
        normalized.
    
    Returns:
    covariance_matrix (numpy.ndarray): A 2D array representing the covariance 
        matrix of the ellipsoid. This matrix describes the shape and orientation 
        of the ellipsoid in terms of a multivariate Gaussian distribution, where 
        the variance along each principal axis is given by the square of the 
        corresponding radius.
    """
    # Eigenvalues are the squares of the radii (if the radii represent 
    #standard deviations along the axes)
    eigenvalues = radii**2
    # Construct the diagonal matrix of eigenvalues
    D = np.diag(eigenvalues)
    # Compute the covariance matrix
    covariance_matrix = evecs @ D @ evecs.T
    return covariance_matrix

def covMat_to_ellParamsQ(covM):
    """
    Convert a 2D or 3D covariance matrix to ellipse/ellipsoid parameters.

    Returns
    -------
    2D:
        (eigenvalues, eigenvectors, axes_lengths, theta_deg)
        where theta_deg is the rotation of the major axis in the x-y plane.

    3D:
        (eigenvalues, eigenvectors, axes_lengths, theta_deg, phi_deg)
        where (theta_deg, phi_deg) are spherical angles of the major-axis eigenvector:
          - theta: azimuth in x-y plane from +x toward +y
          - phi: inclination from +z (0..180)
    """
    covM = np.asarray(covM)

    # Eigenvalue decomposition (symmetric PSD -> eigh is appropriate)
    eigenvalues, eigenvectors = np.linalg.eigh(covM)

    # Sort by descending eigenvalue (major -> minor axis)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Semi-axes lengths (1 std dev)
    axes_lengths = np.sqrt(np.maximum(eigenvalues, 0))

    if covM.shape == (2, 2):
        # Rotation angle of the major axis (first eigenvector) in x-y plane
        v = eigenvectors[:, 0]
        theta_deg = np.degrees(np.arctan2(v[1], v[0]))
        return eigenvalues, eigenvectors, axes_lengths, theta_deg

    if covM.shape == (3, 3):
        # Spherical angles of the major axis direction (first eigenvector)
        v = eigenvectors[:, 0]
        vx, vy, vz = v

        # Normalize defensively (should already be unit, but keep robust)
        r = np.linalg.norm(v)
        if r == 0:
            theta_deg = 0.0
            phi_deg = 0.0
        else:
            vx, vy, vz = vx / r, vy / r, vz / r
            theta_deg = np.degrees(np.arctan2(vy, vx))              # azimuth
            phi_deg = np.degrees(np.arccos(np.clip(vz, -1.0, 1.0))) # inclination from +z

        return eigenvalues, eigenvectors, axes_lengths, theta_deg, phi_deg

    raise ValueError("Input must be a 2x2 or 3x3 covariance matrix")

def rotAngle_to_eigenvectors(theta_degrees):
    """
    Construct a 2×2 rotation matrix whose columns represent the eigenvectors
    (principal axis directions) of an ellipse rotated by `theta_degrees`.
    
    The rotation is defined as a counterclockwise rotation from the x-axis by
    `theta_degrees`. When applied to column vectors, this matrix actively rotates
    points in the plane.
    
    Parameters
    ----------
    theta_degrees : float
        Rotation angle in degrees, measured counterclockwise from the x-axis.
    
    Returns
    -------
    R : (2, 2) numpy.ndarray
        Rotation matrix whose columns are the unit eigenvectors corresponding
        to the rotated coordinate axes (e.g., the major and minor axes of an ellipse).
    """
    
    # Convert angle from degrees to radians
    theta = np.radians(theta_degrees)
    
    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    
    return R

def ellParamsQ_to_covMat(a, b, theta):
    """
    Converts the ellipse parameters (semi-major axis a, semi-minor axis b, 
    rotation angle theta, and center (xc, yc)) back into a 2x2 covariance matrix.

    Parameters:
    - a (float): Semi-major axis of the ellipse.
    - b (float): Semi-minor axis of the ellipse.
    - theta (float): Rotation angle of the ellipse in degrees, measured from the 
      x-axis to the semi-major axis in the counter-clockwise direction.

    Returns:
    - covM (numpy.ndarray): The 2x2 covariance matrix.
    """
    # Rotation matrix R(theta)
    R = rotAngle_to_eigenvectors(theta)

    # Diagonal matrix with a^2 and b^2
    D = np.array([[a**2, 0],
                  [0, b**2]])

    # Compute the covariance matrix: Sigma = R * D * R^T
    covM = R @ D @ R.T

    return covM

def UnitCircleGenerate(nTheta):
    """
    Generate a set of points on the unit circle in two dimensions.
    nTheta - number of samples around the azimuthal theta (0 to 2pi)

    Coordinates are returned in an 2 by (nTheta) matrix, with the rows
    being the x, y coordinates of the points.
    """
    
    #generate a unit sphere in 3D
    theta = np.linspace(0,2*math.pi,nTheta)
    rho = 1
    xCoords = rho*np.cos(theta)
    yCoords = rho*np.sin(theta)
    
    #stuff the coordinates into a single nTheta 
    x = np.stack((xCoords, yCoords), axis = 0)
    
    return x

def PointsOnEllipseQ(a, b, theta, xc, yc, nTheta = 200):
    """
    Generates points on an ellipse using parametric equations.
    
    The function scales points from a unit circle to match the given ellipse
    parameters and then rotates the points by the specified angle.

    Parameters:
    - a (float): The semi-major axis of the ellipse.
    - b (float): The semi-minor axis of the ellipse.
    - theta (float): The rotation angle of the ellipse in degrees, measured 
        from the x-axis to the semi-major axis in the counter-clockwise 
        direction.
    - xc (float): The x-coordinate of the center of the ellipse
    - yc (float): The y-coordinate of the center of the ellipse
    - nTheta (int): The number of angular points used to generate the unit 
        circle, which is then scaled to the ellipse. More points will make the
        ellipse appear smoother. Default value is 200.   

    Returns:
    - x_rotated (array): The x-coordinates of the points on the ellipse.
    - y_rotated (array): The y-coordinates of the points on the ellipse.

    """
    #generate points for unit circle
    circle = UnitCircleGenerate(nTheta) #shape: (2,100)
    x_circle, y_circle = circle[0,:], circle[1,:]
    
    #scale the unit circle to the ellipse
    x_ellipse = a * x_circle
    y_ellipse = b * y_circle
    
    #Rotate the ellipse
    angle_rad = np.radians(theta)
    x_rotated = x_ellipse * np.cos(angle_rad) - y_ellipse * np.sin(angle_rad) + xc
    y_rotated = x_ellipse * np.sin(angle_rad) + y_ellipse * np.cos(angle_rad) + yc
    
    return x_rotated, y_rotated

def covMat3D_to_2DsurfaceSlice(covMat):
    """
    This function converts a 3D covariance matrix for ellipsoids into 2D 
    covariance matrices for ellipses that represent the intersection of these 
    ellipsoids with 2D planes.
    
    Args:
    covMat (numpy.ndarray): (ref_size_dim1, ref_size_dim2, ref_size_dim3, 3, 3)
        representing the covariance matrices of ellipsoids in a 3D grid. Each 
        matrix describes the shape of an ellipsoid at a point in the 3D space.
    
    Returns:
    slice_2d_ellipse (numpy.ndarray): (ref_size_dim1, ref_size_dim2, ref_size_dim3, 3, 2, 2).
        This contains the covariance matrices of the 2D ellipses resulting 
        from the intersection of the 3D ellipsoids with three orthogonal planes 
        at each point in the 3D space.
    """
    # Dimensions of the input 3D covariance matrix
    #number of sampled reference stimulus of the R, G, B planes
    ref_shape = covMat.shape[:3]
    # Initialize an array to hold the 2D covariance matrices
    slice_2d_ellipse = np.full((*ref_shape, 3, 2, 2), np.nan)
    
    # Iterate over all 3D grid indices using np.ndindex
    for ijk in np.ndindex(ref_shape):
        # Consider each permutation of dimensions 
        #[1,2]: GB plane with collapsed R
        #[0,2]: RB plane with collapsed G
        #[0,1]: RG plane with collapsed B
        for _idx, _idx_varying in zip([0,1,2], [[1, 2],[0, 2],[0, 1]]):
            idx = np.array(_idx_varying)
            # Invert the 3D covariance matrix to get the precision matrix
            precision_matrix = np.linalg.inv(covMat[*ijk])
            # Extract the 2x2 matrix corresponding to the x and y dimensions
            precision_2d = precision_matrix[idx][:,idx]
            # Invert the 2x2 precision matrix to get the covariance 
            # matrix of the 2D ellipse
            slice_2d_ellipse[*ijk,_idx] = np.linalg.inv(precision_2d)
    return slice_2d_ellipse

def convert_2Dcov_to_points_on_ellipse(cov2D, ref_x = 0, ref_y = 0, nTheta = 200):
    """
    Convert a 2D covariance matrix into points on an ellipse, scaled and recentered.
    
    Parameters:
    cov2D : array_like
        2x2 covariance matrix.
    scaler : float, optional
        Scaling factor for the size of the ellipse.
    ref_x : float, optional
        X-coordinate to recenter the ellipse.
    ref_y : float, optional
        Y-coordinate to recenter the ellipse.
    
    Returns:
    tuple
        Two arrays representing the x and y coordinates of the ellipse points.
    """
    
    #axes loength and rotation angle
    _,_,axisLength, rotAngle = covMat_to_ellParamsQ(cov2D)
    #poitns on ellipses
    ell_2d_x, ell_2d_y = PointsOnEllipseQ(*axisLength, 
                                          rotAngle, 
                                          ref_x, 
                                          ref_y, 
                                          nTheta = nTheta
                                          )
    return ell_2d_x, ell_2d_y

def find_inner_outer_contours(ell_params):
    """
    This function computes the union and intersection contours of a set of ellipses.
    
    Input:
    - list_ell_params: A NumPy array of size (N x 5), where N is the number of ellipses (N > 1). 
                       Each row contains the parameters [cx, cy, a, b, theta] for an ellipse:
                       - cx, cy: center coordinates of the ellipse.
                       - a, b: semi-major and semi-minor axes lengths.
                       - theta: rotation angle of the ellipse (in radians).
    
    Output:
    - Two pairs of arrays:
        - xu, yu: arrays representing the x and y coordinates of the union contour (outer boundary).
        - xi, yi: arrays representing the x and y coordinates of the intersection contour (inner boundary).
    
    Raises:
    - ValueError: If the input `list_ell_params` does not have the size (N x 5) where N > 1.
    """
    
    # Check if the input array has the correct shape (N x 5) and N > 1
    if ell_params.ndim != 2 or ell_params.shape[1] != 5 or ell_params.shape[0] <= 1:
        raise ValueError("Input must be a NumPy array of shape (N x 5) where N > 1.")
    
    # Get the number of ellipses from the shape of the input array
    nEll = ell_params.shape[0]
    # Extract parameters for the first ellipse (center x, center y, semi-major axis, semi-minor axis, angle)
    cx1, cy1, a1, b1, theta1 = ell_params[0]
    # Generate points on the first ellipse using a helper function
    ellipse1_x, ellipse1_y = PointsOnEllipseQ(a1, b1, theta1, cx1, cy1)
    ellipse1 = np.column_stack((ellipse1_x, ellipse1_y))
    
    # Convert the points of the first ellipse to a Shapely Polygon object
    poly1 = Polygon(ellipse1)
    
    # Initialize the union and intersection polygons with the first ellipse
    union_poly = poly1
    intersection_poly = poly1
    
    # Iterate over the remaining ellipses in the list
    for i in range(1, nEll):
        # Extract the parameters for the i-th ellipse
        cxi, cyi, ai, bi, thetai = ell_params[i]
        # Generate points for the i-th ellipse
        ellipse_i_x, ellipse_i_y = PointsOnEllipseQ(ai, bi, thetai, cxi, cyi)
        ellipse_i = np.column_stack((ellipse_i_x, ellipse_i_y))
        
        # Convert the i-th ellipse to a Shapely Polygon object
        poly_i = Polygon(ellipse_i)
        # Update the union of all ellipses encountered so far
        union_poly = union_poly.union(poly_i)
        # Update the intersection of all ellipses encountered so far
        intersection_poly = intersection_poly.intersection(poly_i)
    
    # Extract the x and y coordinates of the union polygon's outer contour
    xu, yu = union_poly.exterior.xy
    # Extract the x and y coordinates of the intersection polygon's outer contour
    xi, yi = intersection_poly.exterior.xy
    
    # Return the x and y coordinates of both the union and intersection contours as arrays
    # Output sizes: xu, yu, xi, yi are 1D arrays of the same length, representing coordinates of the contours.
    return np.array(xu), np.array(yu), np.array(xi), np.array(yi)

#%%       
def fit_2d_isothreshold_contour(ref, comp, nTheta = 200, ellipse_scaler = 1.0,
                                flag_force_centered_ref = False, pd_eps = 1e-16):
    """
    Fit an ellipse to 2D isothreshold contour points.
    
    Parameters
    ----------
    ref : array-like, shape (2,)
        Reference stimulus location.
    
    comp : array-like, shape (2, M) or (M, 2)
        Contour points.
    
    nTheta : int, optional
        Number of samples used to render the fitted ellipse curve.
    
    ellipse_scaler : float, optional
        Scales the fitted ellipse and contour points about `ref`:
            pts_scaled = (pts - ref) * ellipse_scaler + ref
    
    flag_force_centered_ref : bool, optional
        If True, constrain the fitted ellipse center to be exactly at `ref`.
        If False, fit center as well (standard unconstrained ellipse fit).
    
    pd_eps : float, optional
        Eigenvalue floor used to keep the constrained quadratic form positive
        definite (prevents degenerate / non-ellipse solutions).
    
    Returns
    -------
    fitEllipse_scaled : ndarray, shape (2, nTheta)
    fitEllipse_unscaled : ndarray, shape (2, nTheta)
    ellipse_params : list
        [xCenter, yCenter, majorAxis, minorAxis, theta_deg]
    """
    ref = np.asarray(ref, dtype=float)
    comp = np.asarray(comp, dtype=float)
    
    if ref.shape != (2,):
        raise ValueError(f"`ref` must have shape (2,), got {ref.shape}")
    
    # Accept (2, M) or (M, 2); convert to (2, M)
    if comp.ndim != 2:
        raise ValueError(f"`comp` must be 2D, got shape {comp.shape}")
    if comp.shape[0] == 2:
        comp_unscaled = comp
    elif comp.shape[1] == 2:
        comp_unscaled = comp.T
    else:
        raise ValueError(f"`comp` must have shape (2, M) or (M, 2), got {comp.shape}")
    
    M = comp_unscaled.shape[1]
    if M < 5:
        raise ValueError(f"Need at least 5 points to fit an ellipse; got {M}")
    
    if not flag_force_centered_ref:
        # ---------------------------------------------------------
        # Unconstrained fit (center is estimated)
        # ---------------------------------------------------------
        ellipse = EllipseModel()
        ok = ellipse.estimate(comp_unscaled.T)  # (M, 2)
        if not ok or ellipse.params is None:
            raise RuntimeError("Ellipse fit failed. Contour points may be degenerate or too noisy.")
    
        xCenter, yCenter, majorAxis, minorAxis, theta_rad = ellipse.params
        theta_deg = np.rad2deg(theta_rad)
    
        # Render ellipse curve
        x_fit, y_fit = PointsOnEllipseQ(
            majorAxis, minorAxis, theta_deg, xCenter, yCenter, nTheta
        )
        fitEllipse_unscaled = np.stack((x_fit, y_fit), axis=0)
    
    else:
        # ---------------------------------------------------------
        # Constrained-center fit: center fixed at `ref`
        # Fit A u^2 + B u v + C v^2 = 1 in centered coordinates.
        # ---------------------------------------------------------
        u = comp_unscaled[0, :] - ref[0]
        v = comp_unscaled[1, :] - ref[1]
    
        D = np.stack([u**2, u*v, v**2], axis=1)  # (M, 3)
        y = np.ones((M,), dtype=float)
    
        p, *_ = np.linalg.lstsq(D, y, rcond=None)
        A, B, C = p
    
        Q = np.array([[A, B/2.0], [B/2.0, C]], dtype=float)
    
        # Eigen-decomposition: Q = V diag(lam) V^T
        lam, V =np.linalg.eigh(Q) #lam sorted ascending
    
        if np.any(lam <= 0):
            raise ValueError("Not an ellipse: quadratic form is not positive "+\
                             "definite (eigenvalues must be > 0).")
        
        #lam_small corresponds to major axis (largest radius)
        lam_small, lam_large = lam[0], lam[1]
        v_major = V[:,0] #eigenvector for smallest eigenvalue
        
        majorAxis = 1 / np.sqrt(lam_small)
        minorAxis = 1 / np.sqrt(lam_large)
        
        th = np.arctan2(v_major[1], v_major[0])
        theta_deg = np.rad2deg(th)
    
        xCenter, yCenter = float(ref[0]), float(ref[1])
    
        # Render ellipse curve from axes + rotation about ref
        t = np.linspace(0, 2*np.pi, nTheta, endpoint=False)
        pts = np.stack([majorAxis * np.cos(t), minorAxis * np.sin(t)], axis=0)  # (2, n)
    
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]], dtype=float)
    
        fitEllipse_unscaled = (R @ pts) + ref[:, None]
    
    # Scale fitted ellipse about ref
    fitEllipse_scaled = (fitEllipse_unscaled - ref[:, None]) * ellipse_scaler + ref[:, None]
    comp_scaled = (comp - ref[:, None]) * ellipse_scaler + ref[:, None] 
    
    ellipse_params = [xCenter, yCenter, majorAxis, minorAxis, theta_deg]
    return fitEllipse_scaled, fitEllipse_unscaled, ellipse_params, comp_scaled
        
def stretch_unit_circle(x, y, ax_length_x, ax_length_y, z=None, ax_length_z=None):
    """
    Stretches unit circle (or sphere) coordinates based on the semi-axis lengths.
    
    This function scales the x and y coordinates of points originally on a unit 
    circle to match the specified semi-axis lengths of an ellipse. If a z-axis 
    value is provided, it scales the z-coordinate as well, effectively transforming 
    a unit sphere into an ellipsoid.
    
    Notes:
    ------
    - When applied to a unit circle (2D case), this function stretches the points 
      into an ellipse by scaling x and y separately.
    - When applied to a unit sphere (3D case), it stretches the points into an 
      ellipsoid by also scaling the z-dimension.
    """
    x_stretched = x * ax_length_x
    y_stretched = y * ax_length_y
    if z is None or ax_length_z is None:
        return x_stretched, y_stretched
    else:
        z_stretched = z * ax_length_z
        return x_stretched, y_stretched, z_stretched

def rotate_relocate_stretched_ellipse(x, y, rot_angle, x0, y0):
    """
    Rotates and relocates an ellipse to a specified center.
    
    This function rotates a set of (x, y) coordinates according to a given rotation 
    angle and then translates them to a new center (x0, y0). It is typically used to 
    transform points that were originally aligned with the principal axes of an 
    ellipse into a rotated and relocated version of that ellipse.
    
    Notes:
    ------
    - The function first constructs a rotation matrix from `rot_angle` using 
      `rotAngle_to_eigenvectors(rot_angle)`, which converts the angle into eigenvectors.
    - The points (x, y) are then rotated using the matrix multiplication `R @ xy`.
    - Finally, the rotated points are translated to `(x0, y0)`.
    
    """
    R = rotAngle_to_eigenvectors(rot_angle)  # Convert rotation angle to eigenvectors (rotation matrix)
    xy = np.vstack((x, y))  # Stack x and y into a 2-row matrix
    rot_xy = R @ xy  # Apply rotation
    reloc_xy = rot_xy + np.array([[x0, y0]]).T  # Translate to new center
    return reloc_xy        

def distance_to_ellipse_boundary(a, b, theta_deg, dx, dy):
    """
    Computes the distance from the center of a rotated ellipse to its boundary 
    in a given direction specified by the vector (dx, dy).
    
    Parameters:
    - a (float): Semi-major axis length of the ellipse
    - b (float): Semi-minor axis length of the ellipse
    - theta_deg (float): Rotation angle of the ellipse in degrees, counterclockwise from the x-axis
    - dx (float): x-component of the direction vector
    - dy (float): y-component of the direction vector
    
    Returns:
    - r (float): Distance from the center to the ellipse boundary along the direction (dx, dy)
    - dx (float, optional): Normalized x-component (only returned if input was not unit length)
    - dy (float, optional): Normalized y-component (only returned if input was not unit length)
    """
    
    # Compute the length of the input vector
    vec_len = np.linalg.norm([dx, dy])
    
    # Normalize the vector if it's not already unit length
    if np.abs(vec_len - 1) > 1e-8:
        print('The input vector [dx, dy] is not a unit vector! Normalizing...')
        dx /= vec_len
        dy /= vec_len
        return_extra = True
    else:
        return_extra = False
    
    # Convert rotation angle from degrees to radians
    theta_rad = np.deg2rad(theta_deg)
    
    # Compute quadratic form coefficients for the rotated ellipse
    A = (np.cos(theta_rad) ** 2) / a**2 + (np.sin(theta_rad) ** 2) / b**2
    B = 2 * (1 / a**2 - 1 / b**2) * np.cos(theta_rad) * np.sin(theta_rad)
    C = (np.sin(theta_rad) ** 2) / a**2 + (np.cos(theta_rad) ** 2) / b**2
    
    # Compute the distance to the ellipse boundary in direction (dx, dy)
    r = 1 / np.sqrt(A * dx**2 + B * dx * dy + C * dy**2)
    
    return (r, dx, dy) if return_extra else r

def symmetric_angle_difference(angle1, angle2):
    """
    Compute the smallest symmetric angular difference between two angles (in degrees),
    accounting for the fact that 0° and 180° are equivalent in ellipse orientation.
    
    Parameters:
        angle1, angle2: arrays or scalars of angles in degrees.
    
    Returns:
        Symmetric angular difference in degrees, bounded between 0 and 90.
    """
    diff = np.abs(angle1 - angle2)
    diff = np.where(diff > 180, 360 - diff, diff)
    diff = np.where(diff > 90, 180 - diff, diff)
    return diff

def compute_radii_scaler_to_reach_targetPC(pC_target, ndims = 2, lb = 2, ub = 3,
                                           nsteps = 100, nz = int(1e4),
                                           flag_visualize = False):
    """
    Computes the optimal scaling factor (radii) to reach a target probability of 
    correct classification (pC_target) between three points in a 2D bivariate 
    Gaussian distribution. 
    
    Parameters:
    - pC_target: Target probability of correct classification.
    - ndims: the dimensionality of the Gaussian distribution (2 or 3)
    - lb: Lower bound of the scaler (radius) range to search over.
    - ub: Upper bound of the scaler (radius) range to search over.
    - nsteps: Number of steps for the scaler search range.
    - nz: Number of samples to generate for the distribution.
    
    Returns:
    - opt_scaler: The scaler that yields the probability closest to the target pC_target.
    - probC[min_idx]: The probability of correct classification at the optimal scaler.
    """
    
    # Define the mean vector and covariance matrix for the bivariate Gaussian distribution
    mean = [0] * ndims
    cov = np.eye(ndims)  
    
    # Generate the scaling factors (radii) to be tested between lb and ub
    z2_scaler = np.linspace(lb, ub, nsteps)
    
    # Initialize an array to store the probability of correct classification for each scaler
    probC = np.full((nsteps), np.nan)
    
    # Loop through each scaling factor to compute the probability of correct classification
    for idx, scaler in enumerate(z2_scaler):            
        # Draw nz samples from the bivariate Gaussian distribution for z0 and z1 
        #(two independent points)
        z0 = np.random.multivariate_normal(mean, cov, nz)
        z1 = np.random.multivariate_normal(mean, cov, nz)
    
        # For z2, apply a center offset based on the current scaler value
        z2_center = np.array([0] *(ndims - 1) + [scaler])
        z2 = np.random.multivariate_normal(mean, cov, nz) + z2_center[None, :]
    
        # Compute pairwise differences between points z0, z1, and z2
        r01 = z0 - z1   
        r02 = z0 - z2   
        r12 = z1 - z2   
    
        # Compute squared Mahalanobis distances (a measure of the distance between points 
        # in Gaussian space) Mahalanobis distance accounts for the covariance of the distribution.
        z0_to_z1 = jnp.sum(r01 * jnp.linalg.solve(cov, r01.T).T, axis=1)
        z0_to_z2 = jnp.sum(r02 * jnp.linalg.solve(cov, r02.T).T, axis=1)
        z1_to_z2 = jnp.sum(r12 * jnp.linalg.solve(cov, r12.T).T, axis=1)
    
        # Compute the difference between z0-to-z1 distance and the minimum of z0-to-z2 and 
        # z1-to-z2 distances
        zdiff = z0_to_z1 - jnp.minimum(z0_to_z2, z1_to_z2)
    
        # Calculate the probability of correct classification as the fraction where zdiff < 0
        probC[idx] = np.sum(zdiff < 0) / nz
    
    # Plot the computed probabilities as a function of the scaling factors
    if flag_visualize: plt.plot(z2_scaler, probC)
    
    # Find the index of the scaling factor closest to the target probability (pC_target)
    min_idx = np.argmin(np.abs(pC_target - probC))
    
    # Retrieve the optimal scaling factor and the corresponding probability
    opt_scaler = z2_scaler[min_idx]
    return opt_scaler, probC[min_idx], probC
        
#%%
class GegenfurtnerEll:
    def convert_ellParamsW_to_covMatDKL(a, b, theta, M_trans):
        """
        Convert ellipse parameters in W space to a covariance matrix in DKL space.
        
        Parameters:
        a (float): Major axis length of the ellipse in W space.
        b (float): Minor axis length of the ellipse in W space.
        theta (float): Angle of rotation of the ellipse in degrees in W space.
        M_trans (ndarray): Transformation matrix to convert W space to DKL space.
        
        Returns:
        covMat_ell_DKL (ndarray): Covariance matrix of the ellipse in DKL space.
        """
        # Convert ellipse parameters (a, b, theta) to covariance matrix in W space
        covMat_ell_W = ellParamsQ_to_covMat(a, b, theta)
        
        # Transform the covariance matrix from W space to DKL space
        covMat_ell_DKL = M_trans @ covMat_ell_W @ M_trans.T
        
        return covMat_ell_DKL
    
    def convert_ellParamsW_to_covMatDKL_v2(a, b, theta, M_trans):
        covMat_ell_W = ellParamsQ_to_covMat(a, b, theta)
        
        W_pts_d1, W_pts_d2 = convert_2Dcov_to_points_on_ellipse(covMat_ell_W)
        W_pts = np.stack((W_pts_d1, W_pts_d2, np.ones(W_pts_d1.shape, dtype=float)))
        DKL_pts = M_trans @ W_pts
        
        *_, ellP2 = fit_2d_isothreshold_contour(np.array([0,0]), 
                                                DKL_pts[:2,:],
                                                flag_force_centered_ref = True
                                                )
        covMat_ell_DKL2 = ellParamsQ_to_covMat(*ellP2[2:])
        return covMat_ell_DKL2

    def stretchingMat_from_covMatDKL(covMat_ell_DKL):
        """
        Compute a stretching matrix that converts an ellipse, represented by a 
        covariance matrix in DKL space, into an ellipse with unit x- and y-axis lengths.
        
        Parameters:
        -----------
        covMat_ell_DKL : ndarray
            A 2x2 covariance matrix representing the shape and orientation of the 
            ellipse in DKL space.
        
        Returns:
        --------
        stretchingMat_DKL_to_unit : ndarray
            A 2x2 transformation matrix that scales the ellipse to have unit lengths 
            along the x- and y-axes.
        stretchingMat_unit_to_DKL : ndarray
            The inverse of stretchingMat_DKL_to_unit, which transforms the normalized ellipse 
            back to its original shape and orientation.
        """
        # Extract variances along x- and y-axes
        sigma_xx = covMat_ell_DKL[0, 0]
        sigma_yy = covMat_ell_DKL[1, 1]
        
        # Construct the scaling matrix to normalize x- and y-axis lengths
        scaling_x = 1 / np.sqrt(sigma_xx)
        scaling_y = 1 / np.sqrt(sigma_yy)
        stretchingMat_DKL_to_unit = np.diag([scaling_x, scaling_y])
        
        # Compute the inverse scaling matrix
        stretchingMat_unit_to_DKL = np.linalg.inv(stretchingMat_DKL_to_unit)
        
        return stretchingMat_DKL_to_unit, stretchingMat_unit_to_DKL

    def normalize_ellipse_axes(stretchingMat_DKL_to_unit, cov_matrix_test):    
        """
        Normalize a covariance matrix of an ellipse using a stretching matrix.
        
        This function uses a precomputed stretching matrix to normalize an ellipse 
        (i.e., scale its axes).
        
        Parameters:
        -----------
        stretchingMat_DKL_to_unit : ndarray
            A 2x2 diagonal transformation matrix that stretches an ellipse into another
            ellipse with unit x- and y-axis lengths.
        cov_matrix_test : ndarray
            A 2x2 covariance matrix representing the ellipse to be normalized.
        
        Returns:
        --------
        scaled_matrix : ndarray
            A 2x2 covariance matrix of the ellipse normalized to have unit axes.
        """
        # Apply the stretching matrix to normalize the covariance matrix.
        # This effectively rescales the ellipse to have unit length along its axes.
        scaled_matrix = stretchingMat_DKL_to_unit @ cov_matrix_test @ stretchingMat_DKL_to_unit.T
        
        return scaled_matrix