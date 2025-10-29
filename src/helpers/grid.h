//
// space.h
//
// Simple vector and matrix classes
//

#ifndef __GRID_H__
#define __GRID_H__

// Includes, system
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <omp.h>

// Includes, project
#include "space.h"

inline double sqr(const double x) {
	return x * x;
}

// https://stackoverflow.com/questions/4578967/cube-sphere-intersection-test
inline bool sphere_cube_intersects(const double circle_x, const double circle_y, const double circle_z, const double circle_r, const double rect_x, const double rect_y, const double rect_z, const double rect_width, const double rect_height, const double rect_depth) {
	const double Bmin[] = {rect_x, rect_y, rect_z};
	const double Bmax[] = {rect_x + rect_width, rect_y + rect_height, rect_z + rect_depth};
	const double C[]    = {circle_x, circle_y, circle_z};

	const double r2 = sqr(circle_r);
	double dmin = 0.00;

	for(int i = 0; i < 3; i++) {
    	if(C[i] < Bmin[i]) dmin += sqr(C[i] - Bmin[i]);
    	else if(C[i] > Bmax[i]) dmin += sqr(C[i] - Bmax[i]);     
 	}

	return dmin <= r2;
}

template <typename double_matrix_type, typename int_matrix_type, typename int_vector_type> 
void build_clouds_hash(int max_node_no, double r, double dx, double dy, double dz, double_matrix_type& nodes, int_matrix_type& clouds, int_vector_type& cloud_lengths, int& max_cloud_length)
{
	std::cout << "Step 0: Calculate bounding box (in parallel)\n";

	// Step 0: Calculate bounding box (in parallel)
	double x0 = nodes(0, 0);
	double x1 = nodes(0, 0);
	double y0 = nodes(0, 1);
	double y1 = nodes(0, 1);
	double z0 = nodes(0, 2);
	double z1 = nodes(0, 2);

	// const int node_no = nodes.size1(); // for space
	const int node_no = nodes.rows(); // for Eigen

#pragma omp parallel for reduction(min: x0, y0, z0) reduction(max: x1, y1, z1)  
	for (int i = 0; i < node_no; i++) {
		const double x = nodes(i, 0);
		const double y = nodes(i, 1);
		const double z = nodes(i, 2);

		x0 = std::min(x0, x);
		x1 = std::max(x1, x);
		y0 = std::min(y0, y);
		y1 = std::max(y1, y);
		z0 = std::min(z0, z);
		z1 = std::max(z1, z);
	}

	// Step 1: Build hashes (in parallel)
	std::cout << "Step 1: Build hashes (in parallel)\n";

	std::vector<std::pair<int, int> > hashes(node_no);

	const int multiplier_x = static_cast<const int>((x1 - x0) / dx) + 1;
	const int multiplier_y = static_cast<const int>((y1 - y0) / dy) + 1;

#pragma omp parallel for
	for (int i = 0; i < node_no; i++) {
		hashes[i].first = i;
		hashes[i].second = static_cast<int>((nodes(i, 0) - x0) / dx) + multiplier_x * static_cast<int>((nodes(i, 1) - y0) / dy) + multiplier_x * multiplier_y * static_cast<int>((nodes(i, 2) - z0) / dz);  // Note: returned hash cannot be less than zero
	}

	// Step 2: Sort the hashes (parallel if -D_GLIBCXX_PARALLEL -fopenmp is given to compiler)
	std::cout << "Step 2: Sort the hashes (parallel if -D_GLIBCXX_PARALLEL -fopenmp is given to compiler)\n";

	std::sort(hashes.begin(), hashes.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second < b.second; });

	// Step 3: Build the index
	std::cout << "Step 3: Build the index\n";

	int max_hash = hashes[node_no - 1].second + 1;

	std::vector<int> index(max_hash + 1);
	std::vector<int> size(max_hash);

	std::fill(index.begin(), index.end(), -1);
	std::fill(size.begin(), size.end(), 0);

	int current = -1;
	for (int i = 0; i < node_no; i++) {
		if (current == -1 || hashes[i].second > current) {
			current = hashes[i].second;
			index[current] = i;
		}
	}
	index[max_hash] = node_no;

	int last = 0;
	for (int i = 0; i < max_hash; i++) {
		if (index[i] == -1) {
			size[i] = 0;
		} else {
			int j = i + 1;
			while (index[j] == -1) {
				j++;
			}
			size[i] = index[j] - last;
			last = index[j];
		}
	}

	// Step 4: Calculate bucket corners
	std::cout << "Step 4: Calculate bucket corners\n";

	double_matrix_type corners;
	corners.resize(max_hash, 3);

	#pragma omp parallel for
	for (int i = 0; i < max_hash; i++) {
		int r = (i % (multiplier_x * multiplier_y)) % multiplier_x;
		int c = (i % (multiplier_x * multiplier_y)) / multiplier_x;
		int p = i / (multiplier_x * multiplier_y);
		corners(i, 0) = x0 + r * dx;
		corners(i, 1) = y0 + c * dy;
		corners(i, 2) = z0 + p * dz;
	}

	// Step 5: Build clouds (in parallel)
	std::cout << "Step 5: Build clouds (in parallel)\n";

	clouds.resize(node_no, max_node_no);
	cloud_lengths.resize(node_no);

	// clouds.clear(-1); 		// for space
	clouds.setConstant(-1);	// for Eigen

	std::vector<std::vector<int> > collisions(omp_get_max_threads());
	for (int i = 0; i < omp_get_max_threads(); i++) {
		collisions[i].reserve(max_hash);
	}


	#pragma omp parallel for
	for (int i = 0; i < node_no; i++) {

		int thread = omp_get_thread_num();
		std::vector<int> &local_collisions = collisions[thread];
		local_collisions.clear();
		for (int j = 0; j < max_hash; j++) {
			if (sphere_cube_intersects(nodes(i, 0), nodes(i, 1), nodes(i, 2), r, corners(j, 0), corners(j, 1), corners(j, 2), dx, dy, dz)) {
				local_collisions.push_back(j);
			}
		}


		int fill = 1;
		clouds(i, 0) = i;

		const double x = nodes(i, 0);
		const double y = nodes(i, 1);
		const double z = nodes(i, 2);


		for (int j = 0; j < local_collisions.size(); j++) {
			
			int m = local_collisions[j];
			if (m == -1 || size[m] == 0) {
				continue;
			}
			for (int k = index[m]; k < index[m] + size[m]; k++) {
				int candidate = hashes[k].first;
				if (sqr(x - nodes(candidate, 0)) + sqr(y - nodes(candidate, 1)) + sqr(z - nodes(candidate, 2)) <= sqr(r) && candidate != i) {
					if (fill == max_node_no) {
						throw std::runtime_error("Not enough room to store clouds. Increase max_node_no.");
					}
					clouds(i, fill) = candidate;
					fill++;
				}
			}
		}
		cloud_lengths[i] = fill;
	}

	max_cloud_length = 0;

	#pragma omp parallel for reduction(max: max_cloud_length)
	for (int i = 0; i < node_no; i++) {
		max_cloud_length = std::max(max_cloud_length, static_cast<int>(cloud_lengths[i]));
	}
}

#endif  // __GRID_H__
