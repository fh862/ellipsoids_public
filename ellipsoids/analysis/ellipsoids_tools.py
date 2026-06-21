#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:26:03 2024

@author: fangfang
"""

import numpy as np
from analysis.ellipses_tools import covMat_to_ellParamsQ

#%%
def UnitCircleGenerate_3D(nTheta, nPhi):
    """
    Generates points on the surface of a unit sphere (3D ellipsoid with equal radii) 
    by sampling angles theta and phi in spherical coordinates.
    
    Parameters:
    - nTheta (int): The number of points to sample along the theta dimension.
    - nPhi (int): The number of points to sample along the phi dimension.
            Determines the resolution from top (north pole) to bottom (south pole).
            
    Returns:
    - ellipsoids: A 3D numpy array of shape (nPhi, nTheta, 3), where each "slice" 
        of the array ([..., 0], [..., 1], [..., 2]) corresponds to the x, y, and z 
        coordinates of points on the unit sphere. The first two dimensions 
        correspond to the grid defined by the phi and theta angles, and the 
        third dimension corresponds to the Cartesian coordinates.
    """
    # Generate linearly spaced angles for theta and phi
    theta = np.linspace(0, 2*np.pi, nTheta)
    phi = np.linspace(0, np.pi, nPhi)
    
    # Create 2D grids for theta and phi using meshgrid.
    # THETA and PHI arrays have shapes (nPhi, nTheta) and contain all combinations
    # of phi and theta values.
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Calculate the Cartesian coordinates for points on the unit sphere surface
    # using the spherical to Cartesian coordinate transformation.
    xCoords = np.sin(PHI) * np.cos(THETA)
    yCoords = np.sin(PHI) * np.sin(THETA)
    zCoords = np.cos(PHI)
    
    # Initialize an array to hold the Cartesian coordinates of the points on 
    # the unit sphere. The array is initially filled with NaNs and has the shape 
    #(nPhi, nTheta, 3).
    ellipsoids = np.stack((xCoords, yCoords, zCoords), axis = 2)
    
    return ellipsoids

def rotation_angles_to_eigenvectors(angle1_rad, angle2_rad, angle3_rad):
    """
    Convert three rotation angles into an eigenvector matrix for an ellipsoid.
    
    Parameters
    ----------
    angle1_rad : float
        The first rotation angle in radians (rotation around the z-axis).
        
    angle2_rad : float
        The second rotation angle in radians (rotation around the y-axis).
    
    angle3_rad : float
        The third rotation angle in radians (rotation around the x-axis).
        
    Returns
    -------
    eigenvector_matrix : np.ndarray
        A 3x3 matrix representing the eigenvectors (columns) of the ellipsoid
        after applying the specified rotations.
    """
    # Rotation matrix for a rotation around the x-axis (angle3_rad)
    R_x = np.array([[1,        0,                   0],
                    [0, np.cos(angle3_rad), -np.sin(angle3_rad)],
                    [0, np.sin(angle3_rad),  np.cos(angle3_rad)]])
    
    # Rotation matrix for a rotation around the y-axis (angle2_rad)
    R_y = np.array([[ np.cos(angle2_rad), 0, np.sin(angle2_rad)],
                    [0,                   1,                  0],
                    [-np.sin(angle2_rad), 0, np.cos(angle2_rad)]])
    
    # Rotation matrix for a rotation around the z-axis (angle1_rad)
    R_z = np.array([[np.cos(angle1_rad), -np.sin(angle1_rad), 0],
                    [np.sin(angle1_rad),  np.cos(angle1_rad), 0],
                    [0,                   0,                  1]])
    
    # Combine the rotations: R_x * R_y * R_z
    eigenvector_matrix = R_x @ R_y @ R_z
    
    return eigenvector_matrix

def PointsOnEllipsoid(radii, center, eigenVectors, unitEllipsoid):
    """
    This function computes points on the surface of an ellipsoid given its 
    radii, center, orientation (via eigenVectors), and a unit ellipsoid 
    (unit sphere mapped to an ellipsoid).
    
    Parameters:
    - radii (array; (3,)): Radii of the ellipsoid along the x, y, and z axes.
    - center (array; (3,)): Center of the ellipsoid in 3D space.
    - eigenVectors (array; (3,3)): Rotation matrix representing the orientation 
        of the ellipsoid.
    - unitEllipsoid (array; (nPhi, nTheta,3)): Points on a unit ellipsoid, 
        which is a unit sphere scaled according to the ellipsoid's radii.
                     
    Returns:
    - ellipsoid: A 2D array of size (3, N) containing the 3D coordinates of 
        N points on the ellipsoid's surface. The first dimension corresponds 
        to the x, y, z coordinates, and the second dimension corresponds to 
        the sampled grid points.

    """
    # Extract the x, y, and z coordinates from the unit ellipsoid's surface points.
    x_Ellipsoid = unitEllipsoid[:,:,0]
    y_Ellipsoid = unitEllipsoid[:,:,1]
    z_Ellipsoid = unitEllipsoid[:,:,2]
    
    # Stretch the unit ellipsoid points by the ellipsoid radii to get the 
    # ellipsoid's shape in its principal axis frame.
    x_stretched = x_Ellipsoid * radii[0]
    y_stretched = y_Ellipsoid * radii[1]
    z_stretched = z_Ellipsoid * radii[2]
    
    # Stack the stretched coordinates and flatten them to create a 2D array of size (3, N),
    # where N = Theta * Phi. This step prepares the coordinates for rotation.
    xyz = np.vstack((x_stretched.flatten(), y_stretched.flatten(), z_stretched.flatten()))
    
    # Rotate the stretched ellipsoid points to align with the ellipsoid's actual 
    # orientation in 3D space using the eigenVectors rotation matrix.
    # The resulting xyz_rotated array has size (3, N).
    xyz_rotated = eigenVectors @ xyz
    
    # Translate the rotated points by the ellipsoid's center to position the ellipsoid
    # correctly in 3D space. The size of the ellipsoid array remains (3, N).
    ellipsoid = xyz_rotated + center[:,None]
    
    return ellipsoid

def EllipsoidSurfaceMesh(radii, center, eigenVectors, nu = 120, nv = 240,
                         phi_offset = 0.0):
    """
    Generate a parametric surface mesh (X, Y, Z) for a 3D ellipsoid.

    Parameters
    ----------
    radii : (3,) array_like
        Semi-axis lengths (a, b, c).
    center : (3,) array_like
        Ellipsoid center (x0, y0, z0).
    eigenVectors : (3,3) array_like
        Columns are orthonormal principal axes (rotation matrix).
    nu : int, optional
        # of samples along polar angle θ in [0, π]. Controls vertical resolution.
    nv : int, optional
        # of samples along azimuth φ in [0, 2π]. Controls circumferential resolution.
    phi_offset : float, optional
        Azimuth offset in radians. Useful for rotating the parametric seam
        away from the default front-facing direction without changing the
        ellipsoid geometry.

    Returns
    -------
    X, Y, Z : ndarray, shape (nu, nv)
        Surface coordinates of the ellipsoid. The last azimuth column duplicates
        the first to close the seam at φ=0/2π (useful for some renderers).
    """
    # Affine map: unit-sphere → axis-aligned ellipsoid (scale) → world frame (rotate)
    # p_world = eigenVectors @ diag(radii) @ p_sphere + center
    sphere_to_ellipsoid = np.asarray(eigenVectors, float) @ np.diag(np.asarray(radii, float))

    # Angular parameterization of the unit sphere
    theta = np.linspace(0.0, np.pi,   int(nu))   # polar angle
    phi   = np.linspace(0.0, 2*np.pi, int(nv)) + float(phi_offset)   # azimuth angle
    th, ph = np.meshgrid(theta, phi, indexing="ij")  # (nu, nv)

    # Unit-sphere coordinates s(θ, φ) = (sinθ cosφ, sinθ sinφ, cosθ)
    sphere_pts = np.stack([
        np.sin(th) * np.cos(ph),
        np.sin(th) * np.sin(ph),
        np.cos(th)
    ], axis=-1)  # (nu, nv, 3)

    # Map to world frame and translate by center
    center = np.asarray(center, float).ravel()
    world_pts = sphere_pts @ sphere_to_ellipsoid.T + center  # (nu, nv, 3)

    # Split into component grids
    X, Y, Z = world_pts[..., 0], world_pts[..., 1], world_pts[..., 2]

    # Close the azimuthal seam so column nv-1 == column 0
    X[:, -1], Y[:, -1], Z[:, -1] = X[:, 0], Y[:, 0], Z[:, 0]

    return X, Y, Z

def find_inner_outer_surfaces(cov_mats, center=None, nTheta=240, nPhi=120):
    """
    Compute inner and outer bootstrap-envelope surfaces for common-center
    3D ellipsoids.

    This is the 3D analogue of ``find_inner_outer_contours`` for the common
    reference case. The 2D version can use polygon boolean operations because
    sampled ellipses become simple planar polygons. In 3D, the equivalent
    mesh boolean union/intersection is much more brittle: triangle meshes need
    watertight topology, robust handling of near-tangent intersections, and
    usually a heavier geometry backend. Since all ellipsoids share a center,
    we can instead sample directions on a unit sphere and compute each
    covariance ellipsoid's radial distance along those directions. The outer
    surface uses the maximum radius across ellipsoids; the inner surface uses
    the minimum radius.

    Parameters
    ----------
    cov_mats : array_like
        Covariance matrices for N bootstrap fits. Shape must be (N, 3, 3), or
        any sequence of 3 x 3 covariance matrices. Each matrix defines the
        ellipsoid (x - center)^T inv(cov) (x - center) = 1.

    center : array_like, optional
        Common reference / center for all ellipsoids. If None, the origin is
        used.

    nTheta : int, optional
        Number of azimuth samples on the unit sphere.

    nPhi : int, optional
        Number of polar samples on the unit sphere.

    Returns
    -------
    xu, yu, zu : ndarray, shape (nPhi, nTheta)
        Outer envelope surface coordinates.

    xi, yi, zi : ndarray, shape (nPhi, nTheta)
        Inner envelope surface coordinates.
    """
    cov_mats = np.asarray(cov_mats, dtype=float)
    if cov_mats.ndim != 3 or cov_mats.shape[-2:] != (3, 3) or cov_mats.shape[0] < 1:
        raise ValueError("Input must have shape (N, 3, 3), where N >= 1.")

    if center is None:
        center = np.zeros(3)
    else:
        center = np.asarray(center, dtype=float).reshape(3,)

    dirs = UnitCircleGenerate_3D(nTheta, nPhi)
    dirs_flat = dirs.reshape(-1, 3)
    radii_by_ellipsoid = np.full((cov_mats.shape[0], dirs_flat.shape[0]), np.nan)

    for i, cov in enumerate(cov_mats):
        if not np.allclose(cov, cov.T):
            raise ValueError(f"Covariance matrix at index {i} is not symmetric.")
        eigenvalues, *_ = covMat_to_ellParamsQ(cov)
        if np.any(eigenvalues <= 0):
            raise ValueError(f"Covariance matrix at index {i} is not positive definite.")
        Q = np.linalg.inv(cov)
        denom = np.einsum("ni,ij,nj->n", dirs_flat, Q, dirs_flat)
        if np.any(denom <= 0):
            raise ValueError("Encountered a non-positive ellipsoid radial denominator.")
        radii_by_ellipsoid[i] = 1.0 / np.sqrt(denom)

    r_outer = np.max(radii_by_ellipsoid, axis=0).reshape(dirs.shape[:2])
    r_inner = np.min(radii_by_ellipsoid, axis=0).reshape(dirs.shape[:2])

    outer_pts = center[None, None, :] + r_outer[..., None] * dirs
    inner_pts = center[None, None, :] + r_inner[..., None] * dirs

    xu, yu, zu = outer_pts[..., 0], outer_pts[..., 1], outer_pts[..., 2]
    xi, yi, zi = inner_pts[..., 0], inner_pts[..., 1], inner_pts[..., 2]

    return xu, yu, zu, xi, yi, zi

def ellipsoid_fit(X, lambda_reg = 0):
    """
    This function is taken from
    # http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
    # for arbitrary axes
    See more documentation in the link
    """    
    x=X[:,0]
    y=X[:,1]
    z=X[:,2]
    D = np.array([x*x,
                 y*y,
                 z*z,
                 2 * x*y,
                 2 * x*z,
                 2 * y*z,
                 2 * x,
                 2 * y,
                 2 * z])
    DT = D.conj().T

    v = np.linalg.solve( D.dot(DT)  + lambda_reg * np.eye(D.shape[0]), D.dot(np.ones(np.size(x))))
    A = np.array(  [[v[0], v[3], v[4], v[6]],
                    [v[3], v[1], v[5], v[7]],
                    [v[4], v[5], v[2], v[8]],
                    [v[6], v[7], v[8], -1]])

    center = np.linalg.solve(- A[:3,:3], [[v[6]],[v[7]],[v[8]]])
    T = np.eye(4)
    T[3,:3] = center.T
    R = T.dot(A).dot(T.conj().T)
    evals, evecs = np.linalg.eig(R[:3,:3] / -R[3,3])
    radii = np.sqrt(1. / evals)

    # calculate difference of the fitted points from the actual data normalized by the conic radii
    sgns = np.sign(evals);
    radii = radii * sgns;
    d = np.array([x - center[0], y - center[1], z - center[2]]); # shift data to origin
    d = np.asarray(np.matrix(d.T) * np.matrix(evecs)); # rotate to cardinal axes of the conic;
    d = np.array([d[:,0] / radii[0], d[:,1] / radii[1], d[:,2] / radii[2]]).T; # normalize to the conic radii
    chi2 = np.sum(np.abs(1 - np.sum(d**2 * np.tile(sgns, (d.shape[0], 1)), axis=1)));

    return center, radii, evecs, v, chi2

def ellipsoid_fit_fixed_center(X, center0, lambda_reg=0.0):
    """
    Fit an ellipsoid with its center fixed at center0 = (x0, y0, z0).

    Model in shifted coords u = x - center0:
        u^T Q u = 1
    where Q is symmetric positive definite (ideally).

    Returns:
        center (3,)
        radii  (3,)
        evecs  (3,3)  columns are principal axes
        Q      (3,3)  quadratic form matrix in shifted coords
        chi2   scalar  (same style as your original: sum |1 - u^T Q u|)
    """
    center0 = np.asarray(center0, dtype=float).reshape(3,)
    U = X - center0[None, :]  # shifted data, shape (N, 3)

    x, y, z = U[:, 0], U[:, 1], U[:, 2]

    # Design for symmetric Q (6 params): [xx, yy, zz, 2xy, 2xz, 2yz]
    D = np.vstack([
        x * x,
        y * y,
        z * z,
        2 * x * y,
        2 * x * z,
        2 * y * z
    ])  # shape (6, N)

    # Solve: minimize || D^T p - 1 ||^2 + lambda||p||^2
    # Normal equations in your style: (D D^T + λI) p = D 1
    A = D @ D.T + lambda_reg * np.eye(6)
    b = D @ np.ones(D.shape[1])
    p = np.linalg.solve(A, b)

    # Build Q from p
    Q = np.array([
        [p[0], p[3], p[4]],
        [p[3], p[1], p[5]],
        [p[4], p[5], p[2]]
    ], dtype=float)

    # Eigen-decompose: Q = V diag(evals) V^T
    evals, evecs = np.linalg.eigh(Q)

    # Radii: u^T Q u = 1 => along eigenvector i: evals[i] * u_i^2 = 1 => radius = 1/sqrt(evals[i])
    radii = 1.0 / np.sqrt(evals)

    # Fit error in the same spirit as your chi2
    quad = np.sum((U @ Q) * U, axis=1)  # u^T Q u for each point
    chi2 = np.sum(np.abs(1.0 - quad))

    return center0, radii, evecs, Q, chi2

def fit_3d_isothreshold_ellipsoid(ref, comp, nTheta = 200, nPhi = 100, 
                                  ellipsoid_scaler = 1,
                                  flag_force_centered_ref = False):
    """
    Fit a 3D ellipsoid to a set of comparison stimuli around a reference stimulus,
    then optionally scale the fitted ellipsoid and the discrete comparison points
    radially about the reference.

    Parameters
    ----------
    ref : array-like, shape (3,)
        Reference stimulus (in the same coordinate space as `comp`).
    comp : array-like, shape (N, 3) or (nPhi, nTheta, 3)
        Comparison stimuli to fit. If provided as a grid (nPhi, nTheta, 3),
        it will be flattened to (N, 3) for fitting.
    nTheta : int
        Number of azimuth samples used to generate a unit ellipsoid mesh for visualization.
    nPhi : int
        Number of elevation samples used to generate a unit ellipsoid mesh for visualization.
    ellipsoid_scaler : float
        Radial scaling factor applied about `ref` to both:
        (1) the fitted ellipsoid surface points, and (2) the discrete comparison points.
    flag_force_centered_ref : bool
        If True, constrain the ellipsoid center to be exactly at the reference.
        If False, fit a free-center ellipsoid.

    Returns
    -------
    fitEllipsoid_scaled : ndarray, shape (3, nPhi*nTheta)
        Surface points of the fitted ellipsoid after radial scaling about `ref`.
    fitEllipsoid_unscaled : ndarray, shape (3, nPhi*nTheta)
        Surface points of the fitted ellipsoid before scaling.
    ellFits : dict
        Fitted ellipsoid parameters with keys:
        'center', 'radii', 'evecs', 'v', 'chi2'.
    comp_scaled : ndarray, shape (3, N)
        Discrete comparison points after radial scaling about `ref`.
    """

    # Unit ellipsoid mesh (used only to generate surface points for visualization/output)
    circleIn3D = UnitCircleGenerate_3D(nTheta, nPhi)

    # Accept either a flat list of points (N,3) or a grid (nPhi,nTheta,3).
    if comp.ndim not in (2, 3) or comp.shape[-1] != 3:
        raise ValueError(
            f"`comp` must have shape (N, 3) or (nPhi, nTheta, 3). Got {comp.shape}."
        )

    # Flatten to (N, 3) for fitting
    comp_reshape = comp.reshape(-1, 3)

    # Fit in a reference-centered coordinate system (improves numerical stability):
    # u = comp - ref, so ref is at the origin in u-space.
    comp_centered = comp_reshape - ref[None, :]

    # Fit ellipsoid parameters in centered space; optionally force the center to be exactly 0.
    if flag_force_centered_ref:
        fits = ellipsoid_fit_fixed_center(comp_centered, np.array([0.0, 0.0, 0.0]))
    else:
        fits = ellipsoid_fit(comp_centered)

    # Unpack fit results
    ellFits = {}
    ell_center_centered, ellFits['radii'], ellFits['evecs'], ellFits['v'], ellFits['chi2'] = fits

    # Convert fitted center back to the original coordinate system
    ellFits['center'] = np.asarray(ell_center_centered).reshape(3,) + ref

    # Generate surface points of the fitted ellipsoid (unscaled)
    fitEllipsoid_unscaled = PointsOnEllipsoid(ellFits['radii'],
                                              ellFits['center'],
                                              ellFits['evecs'],
                                              circleIn3D
                                              )

    # Radially scale the ellipsoid surface about the reference
    fitEllipsoid_scaled = (fitEllipsoid_unscaled - ref[:, None]) * ellipsoid_scaler + ref[:, None]

    # Radially scale the discrete comparison points about the reference
    comp_scaled = (comp_reshape.T - ref[:, None]) * ellipsoid_scaler + ref[:, None]

    return fitEllipsoid_scaled, fitEllipsoid_unscaled, ellFits, comp_scaled
      
def eig_to_covMat(eigval, eigvec):
    """
    Compute the covariance matrix of an ellipsoid given eigenvalues and eigenvectors.

    Parameters:
    -----------
    eigval : (N,) array_like
        Eigenvalues (typically variances along each principal axis).
    eigvec : (N, N) array_like
        Eigenvectors as columns (each column is an eigenvector).

    Returns:
    --------
    covMat : (N, N) ndarray
        Covariance matrix reconstructed from eigval and eigvec.
    """
    # Create diagonal matrix of eigenvalues
    Lambda = np.diag(eigval)
    
    # Compute covariance matrix: covMat = eigvec @ Lambda @ eigvec.T
    covMat = eigvec @ Lambda @ eigvec.T
    
    return covMat
          
def slice_ellipsoid_byPlane(center, radii, eigenvectors, plane_v1, plane_v2, 
                            covMat = None, num_grid_pts = 100):
    """
    Computes the intersection of an ellipsoid with a plane, resulting in an elliptical contour.
    
    Parameters:
    - center: A numpy array of shape (3,) representing the center of the ellipsoid.
    - radii: A numpy array of shape (3,) representing the semi-axes (radii) of the ellipsoid.
    - eigenvectors: A numpy array of shape (3, 3) where each column is an eigenvector defining the orientation of the ellipsoid.
    - plane_v1: A numpy array of shape (3,) representing the first vector that lies on the plane.
    - plane_v2: A numpy array of shape (3,) representing the second vector that lies on the plane.
    
    Returns:
    - sliced_ellipse: A numpy array of shape (3, 100) representing the 3D coordinates of the elliptical contour.
    """

    # Normalize the input vectors that define the plane
    v1 = plane_v1 / np.linalg.norm(plane_v1)
    v2 = plane_v2 / np.linalg.norm(plane_v2)
    
    # Check if the two vectors defining the plane are orthogonal
    if np.abs(np.dot(v1, v2)) > 1e-3:  # Allow a small tolerance for floating-point precision
        raise ValueError('The two vectors defining the plane should be orthogonal!')
    
    if covMat is None:
        # Construct the matrix A for the ellipsoid equation
        # The ellipsoid is defined by the equation: (x - center)^T A (x - center) = 1
        # where A = R * D^(-2) * R^T, with R being the rotation matrix (eigenvectors) 
        # and D being the diagonal matrix of radii
        A = eigenvectors @ np.diag(1 / radii**2) @ eigenvectors.T
    else:
        A = covMat

    # Compute the quadratic form in the plane's local coordinate system
    # M is a 2x2 matrix that represents the quadratic form of the ellipsoid equation 
    # restricted to the plane spanned by v1 and v2
    M = np.array([[v1 @ A @ v1, v1 @ A @ v2],
                  [v2 @ A @ v1, v2 @ A @ v2]])

    # Eigendecomposition of M to obtain the semi-axes and rotation of the ellipse
    eigvals, eigvecs = np.linalg.eigh(M)

    # The lengths of the semi-axes of the ellipse are the inverses of the square roots of the eigenvalues
    semi_axes = 1 / np.sqrt(eigvals)
    
    # rotation angle in deg
    rot_angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Parametrize the ellipse in the plane's local coordinate system
    # The ellipse is parameterized using an angle from 0 to 2*pi
    angles = np.linspace(0, 2 * np.pi, num_grid_pts)  # 100 points around the ellipse
    ellipse_local = np.array([semi_axes[0] * np.cos(angles), semi_axes[1] * np.sin(angles)])
    
    # Rotate the ellipse to align with the correct orientation in the plane
    ellipse_local_rotated = eigvecs @ ellipse_local

    # Transform the ellipse from the plane's local coordinates to global 3D coordinates
    # This step places the ellipse in the global coordinate system by using the plane's basis vectors (v1, v2)
    sliced_ellipse = (center[:, None] + 
                      ellipse_local_rotated[0, :] * v1[:, None] + 
                      ellipse_local_rotated[1, :] * v2[:, None])
    
    return sliced_ellipse, [M, eigvals, eigvecs, semi_axes, rot_angle]

def distance_to_ellipsoid_boundary(a, b, c, theta_deg, phi_deg, dx, dy, dz):
    """
    Computes the distance from the center of a rotated ellipsoid to its boundary
    along a direction (dx, dy, dz).

    Ellipsoid model:
      - Axis-aligned ellipsoid: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
      - Rotated so the *major axis* (the 'a' axis) points in direction given by
        spherical angles (theta_deg, phi_deg), where:
          theta: azimuth in x-y plane from +x toward +y (degrees)
          phi:   inclination from +z (degrees), 0..180

    NOTE
    ----
    Using only (theta, phi) fixes the major-axis direction but does NOT fully fix
    the ellipsoid orientation (rotation about the major axis is still free). This
    implementation chooses a canonical rotation with "zero roll".

    Parameters
    ----------
    a, b, c : float
        Semi-axis lengths (a is treated as the axis aligned with (theta, phi) after rotation).
    theta_deg, phi_deg : float
        Orientation angles (degrees) of the 'a' axis in world coordinates.
    dx, dy, dz : float
        Direction vector components.

    Returns
    -------
    r : float
        Distance from center to boundary along the direction.
    (optional) dx, dy, dz : float
        Normalized direction components, only returned if input was not unit length.
    """
    # --- normalize direction if needed ---
    vec_len = np.linalg.norm([dx, dy, dz])
    if np.abs(vec_len - 1) > 1e-8:
        print("The input vector [dx, dy, dz] is not a unit vector! Normalizing...")
        dx, dy, dz = dx / vec_len, dy / vec_len, dz / vec_len
        return_extra = True
    else:
        return_extra = False

    # --- build canonical rotation R that maps local x-axis -> desired major-axis direction ---
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # Desired major-axis unit vector (world coords)
    v = np.array([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ], dtype=float)

    # Pick a reference "up" vector not parallel to v
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    if np.abs(np.dot(v, up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0], dtype=float)

    # Orthonormal basis: e1=v, e2, e3
    e1 = v
    e2 = np.cross(up, e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)

    # Rotation matrix whose columns are the world-space images of local axes
    # local x -> e1 (major axis), local y -> e2, local z -> e3
    R = np.column_stack([e1, e2, e3])

    # Quadratic form Q = R * diag(1/a^2, 1/b^2, 1/c^2) * R^T
    Dinv2 = np.diag([1.0 / a**2, 1.0 / b**2, 1.0 / c**2])
    Q = R @ Dinv2 @ R.T

    d = np.array([dx, dy, dz], dtype=float)
    denom = float(d.T @ Q @ d)

    # Guard against numerical issues (should be > 0 for valid ellipsoid and nonzero d)
    r = 1.0 / np.sqrt(denom)

    return (r, dx, dy, dz) if return_extra else r

def angles_to_3Dchromatic_directions(theta_deg, phi_deg, normalize=True):
    """
    Convert spherical angles (theta, phi) to 3D chromatic direction unit vectors.

    Convention
    ----------
    theta : azimuth in x-y plane, degrees in [0, 360)
    phi   : polar angle from +z axis, degrees in [0, 180]

    Unit vector (before optional renormalization)
    --------------------------------------------
      x = sin(phi) * cos(theta)
      y = sin(phi) * sin(theta)
      z = cos(phi)

    Parameters
    ----------
    theta_deg : (N,) array-like
        Azimuth angles in degrees.
    phi_deg : (N,) array-like
        Polar angles from +z axis in degrees.
    normalize : bool, default True
        If True, renormalize each direction to unit length (safe against
        numerical drift or upstream modifications).

    Returns
    -------
    chromatic_directions : (N, 3) ndarray
        3D direction vectors.
    """
    theta_deg = np.asarray(theta_deg, dtype=float).ravel()
    phi_deg = np.asarray(phi_deg, dtype=float).ravel()

    if theta_deg.shape != phi_deg.shape:
        raise ValueError(f"theta_deg and phi_deg must have the same shape; "
                         f"got {theta_deg.shape} vs {phi_deg.shape}.")

    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)

    chromatic_directions = np.column_stack([
        np.sin(phi_rad) * np.cos(theta_rad),
        np.sin(phi_rad) * np.sin(theta_rad),
        np.cos(phi_rad),
    ])  # (N, 3)

    if normalize:
        n = np.linalg.norm(chromatic_directions, axis=1)
        valid = n > 0
        if not np.all(valid):
            n_bad = np.sum(~valid)
            print(f"[angles_to_chromatic_directions] Warning: {n_bad} direction(s) "
                  "have zero norm; leaving them as zero vectors.")
        chromatic_directions[valid] /= n[valid, None]

    return chromatic_directions

def fibonacci_sphere(n):
    """Generate approximately uniform unit vectors on the sphere.

    Uses a Fibonacci spiral / golden-angle construction to place `n` points
    evenly over the surface of the unit sphere.

    Parameters
    ----------
    n : int
        Number of directions to generate.

    Returns
    -------
    np.ndarray, shape (n, 3)
        Unit vectors `[x, y, z]` distributed approximately uniformly on the sphere.
    """
    i = np.arange(n)

    # Golden angle in radians; spacing successive azimuths by this value
    # avoids clustering and gives an even covering of the sphere.
    phi = np.pi * (3.0 - np.sqrt(5.0))

    # Evenly space points in z so the samples cover the sphere from north
    # pole to south pole without placing points exactly on the poles.
    z = 1 - 2 * (i + 0.5) / n

    # Convert each z-value to the corresponding radius in the x-y plane.
    r = np.sqrt(1 - z**2)

    # Azimuthal angle for each sample along the Fibonacci spiral.
    theta = phi * i

    # Convert cylindrical coordinates to Cartesian coordinates.
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.column_stack([x, y, z])
