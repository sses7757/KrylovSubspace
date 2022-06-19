// This C# class contains matrix eigen and Schur solvers for Hermitian and general matrices.
// This file is a modified copy from another repository of mine (https://github.com/sses7757/Althea)
// which is not yet published in June 2022 and is using GPLv3 license.
// Copyright (C) 2022  https://github.com/sses7757

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Althea.Helpers;
using Althea.Linq;
using Althea.NativeTypes;

namespace Althea.Backend.CSharp.LinearAlgebra;


/// <summary>
/// Static class for solving matrices represented by <see cref="Span{T}"/>s.
/// </summary>
public static unsafe class MatrixSolvers
{
	#region utilities (can be replaced by MKL or CUDA or HIP calls easily)
	#region linear algebra
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static T NormSq<T>(T* vec, int n) where T : unmanaged, IFloatingPoint<T>
	{
		Api.Norm<T>(vec, 1, n, out T norm);
		return norm;
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static T Dot<T>(T* x, T* y, int n) where T : unmanaged, IFloatingPoint<T>
	{
		// true for calculating dot with conjugates
		Api.Inner<T>(true, x, 1, y, 1, n, out T dot);
		return dot;
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static T DotU<T>(T* x, T* y, int n) where T : unmanaged, IFloatingPoint<T>
	{
		// false for calculating dot without conjugates
		Api.Inner<T>(false, x, 1, y, 1, n, out T dot);
		return dot;
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void Scale<T>(T* x, T α, int n) where T : unmanaged, IFloatingPoint<T>
	{
		Api.VectorUnary<T, Api.U_MultiplyScalar>(x, 1, x, 1, n, α);
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void AddScaled<T>(T* y, T* x, T α, int n) where T : unmanaged, IFloatingPoint<T>
	{
		Api.VectorsBinary<T, Api.B_AddScaled>(x, 1, x, 1, y, 1, α, n);
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void AddConjugateScaled<T>(T* y, T* x, T α, int n) where T : unmanaged, IFloatingPoint<T>
	{
		if (NumberType<T>.IsComplex)
			Api.VectorsBinary<T, Api.B_AddConjugateScaled>(x, 1, x, 1, y, 1, α, n);
		else
			AddScaled(y, x, α, n);
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void VecMulMat<T>(T* x, T* A, int ld, T* output, int m, int n) where T : unmanaged, IFloatingPoint<T>
	{
		for (int i = 0; i < n; i++)
		{
			output[i] = Dot(x, A + i * ld, m);
		}
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void VecMulScaledMat<T>(T* A, int ld, T α, T* x, T* output, int m, int n) where T : unmanaged, IFloatingPoint<T>
	{
		for (int i = 0; i < m; i++)
		{
			output[i] = α * Dot(x, A + i * ld, n);
		}
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void MatMulScaledVec<T>(T* A, int ld, T α, T* x, T* output, int m, int n) where T : unmanaged, IFloatingPoint<T>
	{
		Unsafe.InitBlockUnaligned(output, 0, (uint)(m * sizeof(T)));
		if (NumberType<T>.IsComplex)
		{
			for (int i = 0; i < n; i++)
			{
				AddScaled(output, A + i * ld, (x[i] * α).Conjugate(), m);
			}
		}
		else
		{
			for (int i = 0; i < n; i++)
			{
				AddScaled(output, A + i * ld, x[i] * α, m);
			}
		}
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void MatMulVec<T>(T* A, int ld, T* x, T* output, int m, int n) where T : unmanaged, IFloatingPoint<T>
	{
		Unsafe.InitBlockUnaligned(output, 0, (uint)(m * sizeof(T)));
		if (NumberType<T>.IsComplex)
		{
			for (int i = 0; i < n; i++)
			{
				AddScaled(output, A + i * ld, x[i].Conjugate(), m);
			}
		}
		else
		{
			for (int i = 0; i < n; i++)
			{
				AddScaled(output, A + i * ld, x[i], m);
			}
		}
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void Rank1UpdateNeg<T>(T* A, int ld, T* x, T* y, int m, int n, bool conj = true) where T : unmanaged, IFloatingPoint<T>
	{
		if (NumberType<T>.IsComplex && conj)
		{
			for (int i = 0; i < n; i++)
			{
				AddScaled(A + i * ld, x, -y[i].Conjugate(), m);
			}
		}
		else
		{
			for (int i = 0; i < n; i++)
			{
				AddScaled(A + i * ld, x, -y[i], m);
			}
		}
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void SymRank2UpdateNeg<T>(T* A, int ld, T* x, T* y, int n) where T : unmanaged, IFloatingPoint<T>
	{
		if (NumberType<T>.IsComplex)
		{
			for (int i = 0; i < n; i++)
			{
				AddScaled(A + i * ld, x, -y[i].Conjugate(), n);
				AddScaled(A + i * ld, y, -x[i].Conjugate(), n);
			}
		}
		else
		{
			for (int i = 0; i < n; i++)
			{
				AddScaled(A + i * ld, x, -y[i], n);
				AddScaled(A + i * ld, y, -x[i], n);
			}
		}
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void InPlaceTranspose<T>(T* A, int ld, int n) where T : unmanaged, IFloatingPoint<T>
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = i + 1; j < n; j++)
			{
				(A[i + j * ld], A[j + i * ld]) = (A[j + i * ld], A[i + j * ld]);
			}
		}
	}
	#endregion

	#region point wise operations
	private const DataType Accelerated = DataType.RealSingle | DataType.RealDouble | DataType.ComplexSingle | DataType.ComplexDouble;

	// x / (y + scalar)
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void PointWiseDivideAddScalar<T>(T* x, T* y, int n, T scalar, T scalarIm, T* result, T* resultIm) where T : unmanaged, IFloatingPoint<T>
	{
		if (resultIm != null)
		{   // real type
			T imSq = scalarIm * scalarIm;
			scalarIm = -scalarIm;
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			T* xEnd = x + n;
			Vector<T> imSqs = new(imSq), scalarIms = new(scalarIm);
			while (x + Vector<T>.Count <= xEnd)
			{
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var yy = Unsafe.ReadUnaligned<Vector<T>>(y);
				var abs = yy * yy + imSqs;
				yy *= xx / abs;
				xx *= scalarIms / abs;
				Unsafe.WriteUnaligned(result, yy);
				Unsafe.WriteUnaligned(resultIm, xx);
				x += Vector<T>.Count; y += Vector<T>.Count;
				result += Vector<T>.Count; resultIm += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				T abs = y[i] * y[i] + imSq;
				result[i] = x[i] * y[i] / abs;
				resultIm[i] = x[i] * scalarIm / abs;
			}
		}
		else
		{
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			if (NumberType<T>.IsComplex)
			{
				Api.VectorUnary<T, Api.U_AddScalar>(y, 1, result, 1, n, scalar);
				Api.VectorsBinary<T, Api.B_Divide>(x, 1, result, 1, result, 1, default, n);
				return;
			}
			// real type
			T* xEnd = x + n;
			Vector<T> scalars = new(scalar);
			while (x + Vector<T>.Count <= xEnd)
			{
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var yy = Unsafe.ReadUnaligned<Vector<T>>(y);
				xx /= (yy + scalars);
				Unsafe.WriteUnaligned(result, xx);
				x += Vector<T>.Count; y += Vector<T>.Count; result += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				result[i] = x[i] / (y[i] + scalar);
			}
		}
	}
	// x / y
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void PointWiseDivide<T>(T* x, T* y, T* yIm, int n, T* result, T* resultIm) where T : unmanaged, IFloatingPoint<T>
	{
		if (resultIm != null)
		{   // real type
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			T* xEnd = x + n;
			while (x + Vector<T>.Count <= xEnd)
			{
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var yy = Unsafe.ReadUnaligned<Vector<T>>(y);
				var yyI = Unsafe.ReadUnaligned<Vector<T>>(yIm);
				var abs = yy * yy + yyI * yyI;
				var re = xx * yy / abs;
				var im = -xx * yyI / abs;
				Unsafe.WriteUnaligned(result, re);
				Unsafe.WriteUnaligned(resultIm, im);
				x += Vector<T>.Count; y += Vector<T>.Count; yIm += Vector<T>.Count;
				result += Vector<T>.Count; resultIm += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				T abs = y[i] * y[i] + yIm[i] * yIm[i];
				(result[i], resultIm[i]) = (x[i] * y[i] / abs, -x[i] * yIm[i] / abs);
			}
		}
		else
		{
			Api.VectorsBinary<T, Api.B_Divide>(x, 1, y, 1, result, 1, default, n);

		}
	}
	// x * y
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void PointWiseMultiply<T>(T* x, T* xIm, T* y, T* yIm, int n, T* result, T* resultIm) where T : unmanaged, IFloatingPoint<T>
	{
		if (resultIm != null)
		{   // real type
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			T* xEnd = x + n;
			while (x + Vector<T>.Count <= xEnd)
			{
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var xxI = Unsafe.ReadUnaligned<Vector<T>>(xIm);
				var yy = Unsafe.ReadUnaligned<Vector<T>>(y);
				var yyI = Unsafe.ReadUnaligned<Vector<T>>(yIm);
				var re = xx * yy - xxI * yyI;
				var im = xxI * yy + xx * yyI;
				Unsafe.WriteUnaligned(result, re);
				Unsafe.WriteUnaligned(resultIm, im);
				x += Vector<T>.Count; xIm += Vector<T>.Count;
				y += Vector<T>.Count; yIm += Vector<T>.Count;
				result += Vector<T>.Count; resultIm += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				(result[i], resultIm[i]) = (x[i] * y[i] - xIm[i] * yIm[i], xIm[i] * y[i] + x[i] * yIm[i]);
			}
		}
		else
		{
			Api.VectorsBinary<T, Api.B_Multiply>(x, 1, y, 1, result, 1, default, n);
		}
	}
	// x * y + z + scalar
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void PointWiseMultiplyAddScalar<T>(T* x, T* xIm, T* y, T* z, int n, T scalar, T scalarIm, T* result, T* resultIm) where T : unmanaged, IFloatingPoint<T>
	{
		if (resultIm != null)
		{   // real type
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			T* xEnd = x + n;
			Vector<T> scalars = new(scalar), scalarsIm = new(scalarIm);
			while (x + Vector<T>.Count <= xEnd)
			{
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var xxI = Unsafe.ReadUnaligned<Vector<T>>(xIm);
				var yy = Unsafe.ReadUnaligned<Vector<T>>(y);
				var zz = Unsafe.ReadUnaligned<Vector<T>>(z);
				var re = xx * yy + zz + scalars;
				var im = xxI * yy + scalarsIm;
				Unsafe.WriteUnaligned(result, re);
				Unsafe.WriteUnaligned(resultIm, im);
				x += Vector<T>.Count; xIm += Vector<T>.Count;
				y += Vector<T>.Count;
				result += Vector<T>.Count; resultIm += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				(result[i], resultIm[i]) = (x[i] * y[i] + z[i] + scalar, xIm[i] * y[i] + scalarIm);
			}
		}
		else
		{
			Api.VectorsBinary<T, Api.B_Multiply>(x, 1, y, 1, result, 1, default, n);
			Api.VectorsBinary<T, Api.B_Add>(result, 1, z, 1, result, 1, default, n);
			Api.VectorUnary<T, Api.U_AddScalar>(result, 1, result, 1, n, scalar);
		}
	}
	// a * b + x * y
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void PointWiseMultiplyAdd<T>(T* a, T* aIm, T* b, T* x, T* xIm, T* y, int n, T* result, T* resultIm) where T : unmanaged, IFloatingPoint<T>
	{
		if (aIm != null)
		{
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			T* xEnd = x + n;
			while (x + Vector<T>.Count <= xEnd)
			{
				var aa = Unsafe.ReadUnaligned<Vector<T>>(a);
				var aaI = Unsafe.ReadUnaligned<Vector<T>>(aIm);
				var bb = Unsafe.ReadUnaligned<Vector<T>>(b);
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var xxI = Unsafe.ReadUnaligned<Vector<T>>(xIm);
				var yy = Unsafe.ReadUnaligned<Vector<T>>(y);
				Unsafe.WriteUnaligned(result, aa * bb + xx * yy);
				Unsafe.WriteUnaligned(resultIm, aaI * bb + xxI * yy);
				a += Vector<T>.Count; aIm += Vector<T>.Count; b += Vector<T>.Count;
				x += Vector<T>.Count; xIm += Vector<T>.Count; y += Vector<T>.Count;
				result += Vector<T>.Count; resultIm += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				(result[i], resultIm[i]) = (a[i] * b[i] + x[i] * y[i], aIm[i] * b[i] + xIm[i] * y[i]);
			}
		}
		else
		{
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			if (NumberType<T>.IsComplex)
			{
				Api.VectorsBinary<T, Api.B_Multiply>(a, 1, b, 1, resultIm, 1, default, n);
				Api.VectorsBinary<T, Api.B_Multiply>(x, 1, y, 1, result, 1, default, n);
				Api.VectorsBinary<T, Api.B_Add>(result, 1, resultIm, 1, result, 1, default, n);
				return;
			}
			T* xEnd = x + n;
			while (x + Vector<T>.Count <= xEnd)
			{
				var aa = Unsafe.ReadUnaligned<Vector<T>>(a);
				var bb = Unsafe.ReadUnaligned<Vector<T>>(b);
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var yy = Unsafe.ReadUnaligned<Vector<T>>(y);
				Unsafe.WriteUnaligned(result, aa * bb + xx * yy);
				a += Vector<T>.Count; b += Vector<T>.Count;
				x += Vector<T>.Count; y += Vector<T>.Count;
				result += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				result[i] = a[i] * b[i] + x[i] * y[i];
			}
		}
	}
	// 1 / x
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void PointWiseInv<T>(T* x, T* xIm, int n, T* result, T* resultIm) where T : unmanaged, IFloatingPoint<T>
	{
		if (resultIm != null)
		{
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			T* xEnd = x + n;
			while (x + Vector<T>.Count <= xEnd)
			{
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var xxI = Unsafe.ReadUnaligned<Vector<T>>(xIm);
				var abs = xx * xx + xxI * xxI;
				Unsafe.WriteUnaligned(result, xx / abs);
				Unsafe.WriteUnaligned(resultIm, -xxI / abs);
				x += Vector<T>.Count; xIm += Vector<T>.Count;
				result += Vector<T>.Count; resultIm += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				T abs = x[i] * x[i] + xIm[i] * xIm[i];
				(result[i], resultIm[i]) = (x[i] / abs, -xIm[i] / abs);
			}
		}
		else
		{
			Api.VectorUnary<T, Api.U_Reciprocal>(x, 1, result, 1, n, default);
		}
	}
	// x + y * scalar
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void PointWiseAddScaled<T>(T* x, T* xIm, T* y, T* yIm, int n, T scalar, T scalarIm, T* result, T* resultIm) where T : unmanaged, IFloatingPoint<T>
	{
		if (resultIm != null)
		{
			if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
				goto SCALAR;
			T* xEnd = x + n;
			Vector<T> scalars = new(scalar), scalarsIm = new(scalarIm);
			while (x + Vector<T>.Count <= xEnd)
			{
				var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
				var xxI = Unsafe.ReadUnaligned<Vector<T>>(xIm);
				var yy = Unsafe.ReadUnaligned<Vector<T>>(y);
				var yyI = Unsafe.ReadUnaligned<Vector<T>>(yIm);
				Unsafe.WriteUnaligned(result, xx + scalars * yy - scalarIm * yyI);
				Unsafe.WriteUnaligned(resultIm, xxI + scalarsIm * yy + scalars * yyI);
				x += Vector<T>.Count; xIm += Vector<T>.Count;
				y += Vector<T>.Count; yIm += Vector<T>.Count;
				result += Vector<T>.Count; resultIm += Vector<T>.Count;
			}
			n = (int)(xEnd - x);
		SCALAR:
			for (int i = 0; i < n; i++)
			{
				(result[i], resultIm[i]) = (x[i] + scalar * y[i] - scalarIm * yIm[i], xIm[i] + scalarIm * y[i] + scalar * yIm[i]);
			}
		}
		else
		{
			Api.VectorsBinary<T, Api.B_AddScaled>(x, 1, y, 1, result, 1, scalar, n);
		}
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static bool AllZeroComparedTo<T>(T* x, int n, T scalar) where T : unmanaged, IFloatingPoint<T>
	{
		if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
			goto SCALAR;
		T* xEnd = x + n;
		Vector<T> scalars = new(scalar);
		while (x + Vector<T>.Count <= xEnd)
		{
			var xx = Unsafe.ReadUnaligned<Vector<T>>(x);
			xx += scalars;
			if (!Vector.EqualsAll(xx, scalars))
				return false;
			x += Vector<T>.Count;
		}
		n = (int)(xEnd - x);
	SCALAR:
		for (int i = 0; i < n; i++)
		{
			if (x[i] + scalar != scalar)
				return false;
		}
		return true;
	}
	#endregion

	#region orthogonalize
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static (T Re, T Im) Dot<T>(T* xRe, T* xIm, T* yRe, T* yIm, int n) where T : unmanaged, IFloatingPoint<T>
	{
		T dotRe = T.Zero, dotIm = T.Zero;
		if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
			goto SCALAR;
		T* xEnd = xRe + n;
		Vector<T> dotsRe = new(T.Zero), dotsIm = new(T.Zero);
		while (xRe + Vector<T>.Count <= xEnd)
		{
			var xxR = Unsafe.ReadUnaligned<Vector<T>>(xRe);
			var xxI = Unsafe.ReadUnaligned<Vector<T>>(xIm);
			var yyR = Unsafe.ReadUnaligned<Vector<T>>(yRe);
			var yyI = Unsafe.ReadUnaligned<Vector<T>>(yIm);
			dotsRe += xxR * yyR + xxI * yyI;
			dotsIm += xxR * yyI - xxI * yyR;
			xRe += Vector<T>.Count; xIm += Vector<T>.Count;
			yRe += Vector<T>.Count; yIm += Vector<T>.Count;
		}
		n = (int)(xEnd - xRe);
		dotRe = Vector.Sum(dotsRe); dotIm = Vector.Sum(dotsIm);
	SCALAR:
		for (int i = 0; i < n; i++)
		{
			dotRe += xRe[i] * yRe[i] + xIm[i] * yIm[i];
			dotIm += xRe[i] * yIm[i] - xIm[i] * yRe[i];
		}
		return (dotRe, dotIm);
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static T NormSq<T>(T* vecRe, T* vecIm, int n) where T : unmanaged, IFloatingPoint<T>
	{
		T norm = T.Zero;
		if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
			goto SCALAR;
		T* xEnd = vecRe + n;
		Vector<T> norms = new(T.Zero);
		while (vecRe + Vector<T>.Count <= xEnd)
		{
			var xxR = Unsafe.ReadUnaligned<Vector<T>>(vecRe);
			var xxI = Unsafe.ReadUnaligned<Vector<T>>(vecIm);
			norms += xxR * xxR + xxI * xxI;
			vecRe += Vector<T>.Count; vecIm += Vector<T>.Count;
		}
		n = (int)(xEnd - vecRe);
		norm = Vector.Sum(norms);
	SCALAR:
		for (int i = 0; i < n; i++)
			norm += vecRe[i] * vecRe[i] + vecIm[i] * vecIm[i];
		return norm;
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void Scale<T>(T* xRe, T* xIm, int n, T scalar) where T : unmanaged, IFloatingPoint<T>
	{
		Api.VectorUnary<T, Api.U_MultiplyScalar>(xRe, 1, xRe, 1, n, scalar);
		Api.VectorUnary<T, Api.U_MultiplyScalar>(xIm, 1, xIm, 1, n, scalar);
	}
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void Orthogonalize<T>(bool complex, int n, int nVecs, T* Q, int ldq) where T : unmanaged, IFloatingPoint<T>
	{
		if (complex)
		{
			for (int i = 0; i < nVecs; i++)
			{
				T* rRe = Q + 2 * i * ldq, rIm = Q + (2 * i + 1) * ldq;
				for (int j = 0; j < i; j++)
				{
					T* qRe = Q + 2 * j * ldq, qIm = Q + (2 * j + 1) * ldq;
					var (wRe, wIm) = Dot(qRe, qIm, rRe, rIm, n);
					var denom = NormSq(qRe, qIm, n);
					wRe /= denom; wIm /= denom;
					PointWiseAddScaled(rRe, rIm, qRe, qIm, n, -wRe, -wIm, rRe, rIm);
				}
				Scale(rRe, rIm, n, T.One / T.Sqrt(NormSq(rRe, rIm, n)));
			}
		}
		else
		{
			for (int i = 0; i < nVecs; i++)
			{
				var r = Q + i * ldq;
				for (int j = 0; j < i; j++)
				{
					var q = Q + j * ldq;
					var weight = Dot(q, r, n) / NormSq(q, n);
					AddScaled(r, q, -weight, n);
				}
				Scale(r, T.One / T.Sqrt(NormSq(r, n)), n);
			}
		}
	}
	#endregion
	#endregion


	#region orthogonal transformations (can be replaced by MKL or CUDA or HIP calls easily)
	[StructLayout(LayoutKind.Sequential)]
	private readonly ref struct Householder2<T> where T : unmanaged, IFloatingPoint<T>
	{
		private readonly T v1, v2, tau;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Householder2(T v1, T v2)
		{
			T normV = T.Sqrt(v1 * v1 + v2 * v2);
			v1 += T.CopySign(normV, v1);
			this.tau = (T.One + T.One) / (v1 * v1 + v2 * v2);
			this.v1 = v1; this.v2 = v2;
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly Householder2x2<T> ToReflectionMatrix()
		{
			T h2 = T.One - tau * v2 * v2.Conjugate(), h3 = -tau * v1 * v2.Conjugate();
			return new(h2, h3);
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	private readonly ref struct Householder3<T> where T : unmanaged, IFloatingPoint<T>
	{
		private readonly T v1, v2, v3, tau;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Householder3(T v1, T v2, T v3)
		{
			T temp = v2 * v2 + v3 * v3;
			T normV = T.Sqrt(v1 * v1 + temp);
			v1 += T.CopySign(normV, v1);
			this.tau = (T.One + T.One) / (v1 * v1 + temp);
			this.v1 = v1; this.v2 = v2; this.v3 = v3;
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly Householder3x3<T> ToReflectionMatrix()
		{
			T h1 = T.One - tau * v1 * v1.Conjugate(), h2 = T.One - tau * v2 * v2.Conjugate(), h4 = -tau * v1 * v2.Conjugate();
			T h3 = T.One - tau * v3 * v3.Conjugate(), h5 = -tau * v1 * v3.Conjugate(), h6 = -tau * v2 * v3.Conjugate();
			return new(h1, h2, h3, h4, h5, h6);
		}
	}

	//tex:$H = \begin{pmatrix}h_1&h_3\\\bar{h}_3&h_2\end{pmatrix}$
	[StructLayout(LayoutKind.Sequential)]
	private readonly ref struct Householder2x2<T> where T : unmanaged, IFloatingPoint<T>
	{
		private readonly T h1, h2, h3;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Householder2x2(T h22, T h12)
		{
			this.h1 = -h22; this.h2 = h22; this.h3 = h12;
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly void ColumnUpdate(T* A, int ld, int n, T* temp = null)
		{
			if (NumberType<T>.IsComplex)
			{
				Api.VectorUnary<T, Api.U_MultiplyScalar>(A, 1, temp, 1, n, h1);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp, 1, A + ld, 1, temp, 1, h3.Conjugate(), n);
				Api.VectorUnary<T, Api.U_MultiplyScalar>(A + ld, 1, A + ld, 1, n, h2);
				Api.VectorsBinary<T, Api.B_AddScaled>(A + ld, 1, A, 1, A + ld, 1, h3, n);
				Unsafe.CopyBlockUnaligned(A, temp, (uint)(n * sizeof(T)));
			}
			else
			{
				if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
					goto SCALAR;
				T* Aend = A + n;
				T* AA = A + ld;
				while (A + Vector<T>.Count <= Aend)
				{
					var Ai = Unsafe.ReadUnaligned<Vector<T>>(A);
					var Aj = Unsafe.ReadUnaligned<Vector<T>>(AA);
					Unsafe.WriteUnaligned(A, Ai * h1 + Aj * h3);
					Unsafe.WriteUnaligned(AA, Ai * h3 + Aj * h2);
					A += Vector<T>.Count; AA += Vector<T>.Count;
				}
			SCALAR:
				for (int i = 0, j = ld; i < n; i++, j++)
				{
					(A[i], A[j]) = (A[i] * h1 + A[j] * h3, A[i] * h3 + A[j] * h2);
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly void RowUpdate(T* A, int ld, int n)
		{
			T h3c = h3.Conjugate();
			for (int nn = 0, i = 0; nn < n; nn++, i += ld)
			{
				int j = i + 1;
				(A[i], A[j]) = (A[i] * h1 + A[j] * h3, A[i] * h3c + A[j] * h2);
			}
		}
	}

	//tex:$H = \begin{pmatrix}h_1&h_4&h_5\\\bar{h}_4&h_2&h_6\\\bar{h}_5&\bar{h}_6&h_3\end{pmatrix}$
	[StructLayout(LayoutKind.Sequential)]
	private readonly ref struct Householder3x3<T> where T : unmanaged, IFloatingPoint<T>
	{
		private readonly T h1, h2, h3, h4, h5, h6;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Householder3x3(T h1, T h2, T h3, T h4, T h5, T h6)
		{
			this.h1 = h1; this.h2 = h2; this.h3 = h3; this.h4 = h4; this.h5 = h5; this.h6 = h6;
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly void ColumnUpdate(T* A, int ld, int n, T* temp = null)
		{
			if (NumberType<T>.IsComplex)
			{
				T* temp1 = temp, temp2 = temp + n;
				Api.VectorUnary<T, Api.U_MultiplyScalar>(A, 1, temp1, 1, n, h1);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp1, 1, A + ld, 1, temp1, 1, h4.Conjugate(), n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp1, 1, A + 2 * ld, 1, temp1, 1, h5.Conjugate(), n);

				Api.VectorUnary<T, Api.U_MultiplyScalar>(A, 1, temp2, 1, n, h4);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp2, 1, A + ld, 1, temp2, 1, h2, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp2, 1, A + 2 * ld, 1, temp2, 1, h6.Conjugate(), n);

				Api.VectorUnary<T, Api.U_MultiplyScalar>(A + 2 * ld, 1, A + 2 * ld, 1, n, h3);
				Api.VectorsBinary<T, Api.B_AddScaled>(A + 2 * ld, 1, A, 1, A + 2 * ld, 1, h5, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(A + 2 * ld, 1, A + ld, 1, A + 2 * ld, 1, h6, n);

				Unsafe.CopyBlockUnaligned(A, temp1, (uint)(n * sizeof(T)));
				Unsafe.CopyBlockUnaligned(A + ld, temp2, (uint)(n * sizeof(T)));
			}
			else
			{
				if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
					goto SCALAR;
				T* Aend = A + n;
				T* AA = A + ld, AAA = A + 2 * ld;
				while (A + Vector<T>.Count <= Aend)
				{
					var Ai = Unsafe.ReadUnaligned<Vector<T>>(A);
					var Aj = Unsafe.ReadUnaligned<Vector<T>>(AA);
					var Ak = Unsafe.ReadUnaligned<Vector<T>>(AAA);
					Unsafe.WriteUnaligned(A,   Ai * h1 + Aj * h4 + Ak * h5);
					Unsafe.WriteUnaligned(AA,  Ai * h4 + Aj * h2 + Ak * h6);
					Unsafe.WriteUnaligned(AAA, Ai * h5 + Aj * h6 + Ak * h3);
					A += Vector<T>.Count; AA += Vector<T>.Count; AAA += Vector<T>.Count;
				}
			SCALAR:
				for (int i = 0, j = ld, k = 2 * ld; i < n; i++, j++, k++)
				{
					(A[i], A[j], A[k]) =
					(
						A[i] * h1 + A[j] * h4 + A[k] * h5,
						A[i] * h4 + A[j] * h2 + A[k] * h6,
						A[i] * h5 + A[j] * h6 + A[k] * h3
					);
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly void RowUpdate(T* A, int ld, int n)
		{
			if (!Vector.IsHardwareAccelerated || NumberType<T>.IsComplex || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16 || Vector<T>.Count < 3)
			{   // scalar
				T h4c = h4.Conjugate(), h5c = h5.Conjugate(), h6c = h6.Conjugate();
				for (int i = 0, j = 0; i < n; i++, j += ld)
				{
					(A[j], A[j + 1], A[j + 2]) =
					(
						A[j] * h1 + A[j + 1] * h4 + A[j + 2] * h5,
						A[j] * h4c + A[j + 1] * h2 + A[j + 2] * h6,
						A[j] * h5c + A[j + 1] * h6c + A[j + 2] * h3
					);
				}
			}
			else
			{   // vector
				Span<T> vals = stackalloc T[Vector<T>.Count];
				vals.Fill(T.Zero);
				vals[0] = h1; vals[1] = h4; vals[2] = h5;
				Vector<T> h145 = new(vals);
				vals[0] = h4; vals[1] = h2; vals[3] = h6;
				Vector<T> h426 = new(vals);
				vals[0] = h5; vals[1] = h6; vals[3] = h3;
				Vector<T> h563 = new(vals);
				var maskBytes = vals.As<T, byte>();
				for (int i = 0; i < 3 * sizeof(T); i++)
					maskBytes[i] = byte.MaxValue;
				Vector<T> mask = new(maskBytes);
				for (int nn = 0; nn < n; nn++, A += ld)
				{
					var a = Unsafe.ReadUnaligned<Vector<T>>(A);
					// select to remove possible NaNs
					a = Vector.ConditionalSelect(mask, a, h145);
					T Ai = Vector.Sum(a * h145), Aj = Vector.Sum(a * h426), Ak = Vector.Sum(a * h563);
					A[0] = Ai; A[1] = Aj; A[2] = Ak;
				}
			}
		}
	}

	//tex:$Q = \begin{pmatrix}q_1&q_4&q_7\\q_2&q_5&q_8\\q_3&q_6&q_9\end{pmatrix}$
	[StructLayout(LayoutKind.Sequential)]
	private readonly ref struct Orthogonal3x3<T> where T : unmanaged, IFloatingPoint<T>
	{
		private readonly T q1, q2, q3, q4, q5, q6, q7, q8, q9;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public Orthogonal3x3(T q1, T q2, T q3, T q4, T q5, T q6, T q7, T q8, T q9)
		{
			this.q1 = q1; this.q2 = q2; this.q3 = q3; this.q4 = q4; this.q5 = q5;
			this.q6 = q6; this.q7 = q7; this.q8 = q8; this.q9 = q9;
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly void ColumnUpdate(T* A, int ld, int n, T* temp = null)
		{
			if (NumberType<T>.IsComplex)
			{
				T* temp1 = temp, temp2 = temp + n;
				Api.VectorUnary<T, Api.U_MultiplyScalar>(A, 1, temp1, 1, n, q1);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp1, 1, A + ld, 1, temp1, 1, q2, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp1, 1, A + 2 * ld, 1, temp1, 1, q3, n);

				Api.VectorUnary<T, Api.U_MultiplyScalar>(A, 1, temp2, 1, n, q4);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp2, 1, A + ld, 1, temp2, 1, q5, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp2, 1, A + 2 * ld, 1, temp2, 1, q6.Conjugate(), n);

				Api.VectorUnary<T, Api.U_MultiplyScalar>(A + 2 * ld, 1, A + 2 * ld, 1, n, q9);
				Api.VectorsBinary<T, Api.B_AddScaled>(A + 2 * ld, 1, A, 1, A + 2 * ld, 1, q7, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(A + 2 * ld, 1, A + ld, 1, A + 2 * ld, 1, q8, n);

				Unsafe.CopyBlockUnaligned(A, temp1, (uint)(n * sizeof(T)));
				Unsafe.CopyBlockUnaligned(A + ld, temp2, (uint)(n * sizeof(T)));
			}
			else
			{
				if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
					goto SCALAR;
				T* Aend = A + n;
				T* AA = A + ld, AAA = A + 2 * ld;
				while (A + Vector<T>.Count <= Aend)
				{
					var Ai = Unsafe.ReadUnaligned<Vector<T>>(A);
					var Aj = Unsafe.ReadUnaligned<Vector<T>>(AA);
					var Ak = Unsafe.ReadUnaligned<Vector<T>>(AAA);
					Unsafe.WriteUnaligned(A,   Ai * q1 + Aj * q2 + Ak * q3);
					Unsafe.WriteUnaligned(AA,  Ai * q4 + Aj * q5 + Ak * q6);
					Unsafe.WriteUnaligned(AAA, Ai * q7 + Aj * q8 + Ak * q9);
					A += Vector<T>.Count; AA += Vector<T>.Count; AAA += Vector<T>.Count;
				}
			SCALAR:
				for (int i = 0, j = ld, k = 2 * ld; i < n; i++, j++, k++)
				{
					(A[i], A[j], A[k]) =
					(
						A[i] * q1 + A[j] * q2 + A[k] * q3,
						A[i] * q4 + A[j] * q5 + A[k] * q6,
						A[i] * q7 + A[j] * q8 + A[k] * q9
					);
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private readonly void RowUpdateManaged(T* A, int ld, int n)
		{
			for (int nn = 0, i = 0; nn < n; nn++, i += ld)
			{
				int j = i + 1, k = j + 1;
				(A[i], A[j], A[k]) =
				(
					A[i] * q1 + A[j] * q2 + A[k] * q3,
					A[i] * q4 + A[j] * q5 + A[k] * q6,
					A[i] * q7 + A[j] * q8 + A[k] * q9
				);
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly void RowUpdate(T* A, int ld, int n)
		{
			if (!Vector.IsHardwareAccelerated || NumberType<T>.IsComplex || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16 || Vector<T>.Count < 3)
			{   // scalar
				var conj = this;
				if (NumberType<T>.IsComplex)
				{
					Span<T> values = MemoryMarshal.CreateSpan(ref Unsafe.AsRef(in conj.q1), 9);
					for (int i = 0; i < values.Length; i++)
					{
						values[i] = values[i].Conjugate();
					}
				}
				conj.RowUpdateManaged(A, ld, n);
			}
			else
			{   // vector
				Span<T> vals = stackalloc T[Vector<T>.Count];
				vals.Fill(T.Zero);
				vals[0] = q1; vals[1] = q2; vals[2] = q3;
				Vector<T> q123 = new(vals);
				vals[0] = q4; vals[1] = q5; vals[3] = q6;
				Vector<T> q456 = new(vals);
				vals[0] = q7; vals[1] = q8; vals[3] = q9;
				Vector<T> q789 = new(vals);
				var maskBytes = vals.As<T, byte>();
				for (int i = 0; i < 3 * sizeof(T); i++)
					maskBytes[i] = byte.MaxValue;
				Vector<T> mask = new(maskBytes);
				for (int nn = 0; nn < n; nn++, A += ld)
				{
					var a = Unsafe.ReadUnaligned<Vector<T>>(A);
					// select to remove possible NaNs
					a = Vector.ConditionalSelect(mask, a, q123);
					T Ai = Vector.Sum(a * q123), Aj = Vector.Sum(a * q456), Ak = Vector.Sum(a * q789);
					A[0] = Ai; A[1] = Aj; A[2] = Ak;
				}
			}
		}
	}

	[StructLayout(LayoutKind.Sequential)]
	private readonly ref struct Orthogonal4x4<T> where T : unmanaged, IFloatingPoint<T>
	{
		private readonly T q1, q2, q3, q4, q5, q6, q7, q8, q9, qA, qB, qC, qD, qE, qF, qG;

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly void ColumnUpdate(T* A, int ld, int n, T* temp = null)
		{
			T* AA = A + ld, AAA = A + 2 * ld, AAAA = A + 3 * ld;
			if (NumberType<T>.IsComplex)
			{
				T* temp1 = temp, temp2 = temp + n, temp3 = temp + n * 2;
				Api.VectorUnary<T, Api.U_MultiplyScalar>(A, 1, temp1, 1, n, q1);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp1, 1, AA, 1, temp1, 1, q2, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp1, 1, AAA, 1, temp1, 1, q3, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp1, 1, AAAA, 1, temp1, 1, q4, n);

				Api.VectorUnary<T, Api.U_MultiplyScalar>(A, 1, temp2, 1, n, q5);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp2, 1, AA, 1, temp2, 1, q6, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp2, 1, AAA, 1, temp2, 1, q7, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp2, 1, AAAA, 1, temp2, 1, q8, n);

				Api.VectorUnary<T, Api.U_MultiplyScalar>(A, 1, temp3, 1, n, q9);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp3, 1, AA, 1, temp3, 1, qA, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp3, 1, AAA, 1, temp3, 1, qB, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(temp3, 1, AAAA, 1, temp3, 1, qC, n);

				Api.VectorUnary<T, Api.U_MultiplyScalar>(AAAA, 1, AAAA, 1, n, qG);
				Api.VectorsBinary<T, Api.B_AddScaled>(AAAA, 1, A, 1, AAAA, 1, qD, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(AAAA, 1, AA, 1, AAAA, 1, qE, n);
				Api.VectorsBinary<T, Api.B_AddScaled>(AAAA, 1, AAA, 1, AAAA, 1, qF, n);

				Unsafe.CopyBlockUnaligned(A, temp1, (uint)(n * sizeof(T)));
				Unsafe.CopyBlockUnaligned(AA, temp2, (uint)(n * sizeof(T)));
				Unsafe.CopyBlockUnaligned(AAA, temp3, (uint)(n * sizeof(T)));
			}
			else
			{
				if (!Vector.IsHardwareAccelerated || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16)
					goto SCALAR;
				T* Aend = A + n;
				while (A + Vector<T>.Count <= Aend)
				{
					var Ai = Unsafe.ReadUnaligned<Vector<T>>(A);
					var Aj = Unsafe.ReadUnaligned<Vector<T>>(AA);
					var Ak = Unsafe.ReadUnaligned<Vector<T>>(AAA);
					var Al = Unsafe.ReadUnaligned<Vector<T>>(AAAA);
					Unsafe.WriteUnaligned(A,    Ai * q1 + Aj * q2 + Ak * q3 + Al * q4);
					Unsafe.WriteUnaligned(AA,   Ai * q5 + Aj * q6 + Ak * q7 + Al * q8);
					Unsafe.WriteUnaligned(AAA,  Ai * q9 + Aj * qA + Ak * qB + Al * qC);
					Unsafe.WriteUnaligned(AAAA, Ai * qD + Aj * qE + Ak * qF + Al * qG);
					A += Vector<T>.Count; AA += Vector<T>.Count; AAA += Vector<T>.Count; AAAA += Vector<T>.Count;
				}
			SCALAR:
				for (int i = 0, j = ld, k = ld * 2, l = ld * 3; i < n; i++, j++, k++, l++)
				{
					(A[i], A[j], A[k], A[l]) =
					(
						A[i] * q1 + A[j] * q2 + A[k] * q3 + A[l] * q4,
						A[i] * q5 + A[j] * q6 + A[k] * q7 + A[l] * q8,
						A[i] * q9 + A[j] * qA + A[k] * qB + A[l] * qC,
						A[i] * qD + A[j] * qE + A[k] * qF + A[l] * qG
					);
				}
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private readonly void RowUpdateManaged(T* A, int ld, int n)
		{
			for (int nn = 0, i = 0; nn < n; nn++, i += ld)
			{
				int j = i + 1, k = j + 1, l = k + 1;
				(A[i], A[j], A[k], A[l]) =
				(
					A[i] * q1 + A[j] * q2 + A[k] * q3 + A[l] * q4,
					A[i] * q5 + A[j] * q6 + A[k] * q7 + A[l] * q8,
					A[i] * q9 + A[j] * qA + A[k] * qB + A[l] * qC,
					A[i] * qD + A[j] * qE + A[k] * qF + A[l] * qG
				);
			}
		}
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		public readonly void RowUpdate(T* A, int ld, int n)
		{
			if (!Vector.IsHardwareAccelerated || NumberType<T>.IsComplex || (Unmanaged<T>.DataType & Accelerated) == 0 || n < 16 || Vector<T>.Count < 4)
			{   // scalar
				var conj = this;
				if (NumberType<T>.IsComplex)
				{
					Span<T> values = MemoryMarshal.CreateSpan(ref Unsafe.AsRef(in conj.q1), 9);
					for (int i = 0; i < values.Length; i++)
					{
						values[i] = values[i].Conjugate();
					}
				}
				conj.RowUpdateManaged(A, ld, n);
			}
			else
			{   // vector
				Span<T> vals = stackalloc T[Vector<T>.Count];
				vals.Fill(T.Zero);
				vals[0] = q1; vals[1] = q2; vals[2] = q3; vals[3] = q4;
				Vector<T> q1234 = new(vals);
				vals[0] = q5; vals[1] = q6; vals[2] = q7; vals[3] = q8;
				Vector<T> q5678 = new(vals);
				vals[0] = q9; vals[1] = qA; vals[2] = qB; vals[3] = qC;
				Vector<T> q9ABC = new(vals);
				vals[0] = qD; vals[1] = qE; vals[2] = qF; vals[3] = qG;
				Vector<T> qDEFG = new(vals);
				var maskBytes = vals.As<T, byte>();
				for (int i = 0; i < 4 * sizeof(T); i++)
					maskBytes[i] = byte.MaxValue;
				Vector<T> mask = new(maskBytes);
				for (int nn = 0; nn < n; nn++, A += ld)
				{
					var a = Unsafe.ReadUnaligned<Vector<T>>(A);
					if (Vector<T>.Count != 4)
					{
						// select to remove possible NaNs
						a = Vector.ConditionalSelect(mask, a, q1234);
					}
					T Ai = Vector.Sum(a * q1234), Aj = Vector.Sum(a * q5678),
					  Ak = Vector.Sum(a * q9ABC), Al = Vector.Sum(a * qDEFG);
					A[0] = Ai; A[1] = Aj; A[2] = Ak; A[3] = Al;
				}
			}
		}
	}
	#endregion


	#region QR
	/// <summary>
	/// Perform the QR factorization of <paramref name="matrix"/>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="matrix">The input / output matrix to be QR factorized. Its lower triangular part will be replaced Householder reflectors and upper part by the result triangular matrix.</param>
	/// <param name="diagStore">The output <see cref="Span{T}"/> to store the real diagonal elements of the resulting triangular matrix.</param>
	public static void QrFactorize<T>(SpanMatrix<T> matrix, Span<T> diagStore) where T : unmanaged, IFloatingPoint<T>
	{
		if (matrix.Rows <= 1)
			return;
		int m = matrix.Rows, n = matrix.Cols, ld = matrix.LeadDim;
		if (diagStore.Length < Math.Min(m, n))
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(diagStore));
		fixed (T* A = matrix.UnderlyingSpan, diag = diagStore)
		{ 
			// reduce to triangular by Householder reflect from the first column
			int mn = Math.Min(m, n);
			for (int i = 0; i < mn; i++)
			{
				// get vector u and store in A[i:,i]
				//tex:$$\vec{u} = \pmatrix{A_{i,i} \pm \|\vec{A}_{i:,i}\| \\ \vec{A}_{i+1:,i}}$$
				T* u = A + (i + i * ld);
				T normSqU = NormSq(u, m - i), normU = T.Sqrt(normSqU);
				normU = T.CopySign(normU, u[0]);
				// get tau and A[i, i]
				//tex: $$\tau = {2}/{\|\vec{u}\|^2}$$
				T tau = T.One / (normSqU + T.Abs(u[0] * normU));
				//tex:$$H = I - \tau \vec{u}\vec{u}^* $$
				//tex:$$A_{i,i}' = \vec{H}_{i,i:} \vec{A}_{i:,i} = - \|\vec{A}_{i:,i}\|$$
				u[0] += normU;
				diag[i] = -normU;
				Scale(u, T.Sqrt(tau), m - i);
				// get p and store temporarily in diag[(i+1)..]
				//tex:$\vec{p}^* = \vec{u}^* A_{i:,i:}$
				VecMulMat(u, A + (i + (i + 1) * ld), ld, diag + (i + 1), m - i, n - i - 1);
				//tex:$A_{i:,i:} = H A_{i:,i:} = A_{i:,i:} - \tau \vec{u} \vec{p}^*$
				Rank1UpdateNeg(A + (i + (i + 1) * ld), ld, u, diag + (i + 1), m - i, n - i - 1, false);
			}
		}
	}

	/// <summary>
	/// Generate the Q matrix from the output of <see cref="QrFactorize{T}(SpanMatrix{T}, Span{T})"/>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="colLeft">The left most Q matrix's column's index to generate.</param>
	/// <param name="colRight">The right most Q matrix's column's index to generate (exclusive).</param>
	/// <param name="matrix">The output of <see cref="QrFactorize{T}(SpanMatrix{T}, Span{T})"/> whose columns from <paramref name="colLeft"/> to <paramref name="colRight"/> will be overwritten by the result Q matrix.</param>
	/// <param name="workSpace">The working space with length ≥ the number of rows of <paramref name="matrix"/>.</param>
	public static void QrGenerateQ<T>(int colLeft, int colRight, SpanMatrix<T> matrix, Span<T> workSpace) where T : unmanaged, IFloatingPoint<T>
	{
		if (matrix.Rows <= 1)
			return;
		int m = matrix.Rows, n = matrix.Cols, ld = matrix.LeadDim;
		if (workSpace.Length < m)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(workSpace));
		fixed (T* A = matrix.UnderlyingSpan, work = workSpace)
		{   // generate Q starting from the last column that stores vector u
			int mn = Math.Min(m, n) - 1;
			for (int i = mn; i >= 0; i--)
			{
				if (i >= colRight)
					continue;
				int len = m - i;
				// get u from A's column i and copy to work space
				T* u = work, Aii = A + (i + i * ld);
				Unsafe.CopyBlockUnaligned(work, Aii, (uint)(len * sizeof(T)));
				// prepare matrix A[i.., i..]
				if (m > n && i == mn)
				{   // H_n for full-sized Q, all fill with identity matrix
					for (int j = i; j < m; j++)
					{
						if (j < colLeft || j >= colRight)
							continue;
						Unsafe.InitBlockUnaligned(A + (j + 1 + j * ld), 0, (uint)((len - 1) * sizeof(T)));
						A[j + j * ld] = T.One;
					}
				}
				else
				{   // only fill the first row and column of A[i.., i..] for this iteration
					if (i >= colLeft)
					{
						Aii[0] = T.One;
						for (int j = i + 1; j < m; j++)
						{
							A[j + i * ld] = T.Zero;
							if (j >= colLeft && j < colRight)
								A[i + j * ld] = T.Zero;
						}
					}
					else
					{
						for (int j = Math.Max(i + 1, colLeft); j < m && j < colRight; j++)
						{
							A[i + j * ld] = T.Zero;
						}
					}
				}
				// update Householder reflectors' product stored in A[i.., i..]
				//tex:$H_{(i)} = H_{(i-1)} - \tau \vec{u}_{(i)}\vec{u}_{(i)}^* H_{(i-1)}$
				for (int j = Math.Max(i, colLeft); j < m && j < colRight; j++)
				{
					T dot = Dot(u, A + (i + j * ld), len);
					AddScaled(A + (i + j * ld), u, -dot, len);
				}
			}
		}
	}

	/// <summary>
	/// Compute the multiplication of Q matrix's (conjugate) transpose and <paramref name="rightHandSides"/> with output <paramref name="matrix"/> from <see cref="QrFactorize{T}(SpanMatrix{T}, Span{T})"/>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="matrix">The output of <see cref="QrFactorize{T}(SpanMatrix{T}, Span{T})"/>, not modified</param>
	/// <param name="rightHandSides">The input / output matrix to right multiply Q's (conjugate) transpose</param>
	public static void QrQtMultiply<T>(SpanMatrix<T> matrix, SpanMatrix<T> rightHandSides) where T : unmanaged, IFloatingPoint<T>
	{
		if (matrix.Rows <= 1)
			return;
		int m = matrix.Rows, n = matrix.Cols, lda = matrix.LeadDim, nrhs = rightHandSides.Cols, ldb = rightHandSides.LeadDim;
		if (rightHandSides.Rows != m)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(rightHandSides));
		fixed (T* A = matrix.UnderlyingSpan, B = rightHandSides.UnderlyingSpan)
		{
			for (int i = 0; i < n; i++)
			{
				//tex: compute $H_{(i)} B = B - \tau \vec{u}_{(i)}\vec{u}_{(i)}^* B$
				for (int j = 0; j < nrhs; j++)
				{
					T dot = Dot(A + (i + i * lda), B + (i + j * ldb), m - i);
					AddScaled(B + (i + j * ldb), A + (i + i * lda), -dot, m - i);
				}
			}
		}
	}

	/// <summary>
	/// Solve a set of linear equations <c>A * X == B</c> where A is the <paramref name="matrix"/> and B is the <paramref name="rightHandSides"/>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="matrix">The output matrix of <see cref="QrFactorize{T}(SpanMatrix{T}, Span{T})"/>, must be set to <c><see cref="SpanMatrix{T}.Rows"/> == <see cref="SpanMatrix{T}.Cols"/></c>, not modified</param>
	/// <param name="diagStore">The output <see cref="Span{T}"/> of <see cref="QrFactorize{T}(SpanMatrix{T}, Span{T})"/>, not modified</param>
	/// <param name="rightHandSides">The output right hand side matrix of <see cref="QrQtMultiply{T}(SpanMatrix{T}, SpanMatrix{T})"/>, replaced by the solution after return</param>
	public static void QrLinearSolve<T>(SpanMatrix<T> matrix, ReadOnlySpan<T> diagStore, SpanMatrix<T> rightHandSides) where T : unmanaged, IFloatingPoint<T>
	{
		if (matrix.Rows <= 1)
			return;
		int m = matrix.Rows, n = matrix.Cols, lda = matrix.LeadDim, nrhs = rightHandSides.Cols, ldb = rightHandSides.LeadDim;
		if (m != n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(matrix));
		if (rightHandSides.Rows != m)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(rightHandSides));
		fixed (T* A = matrix.UnderlyingSpan, B = rightHandSides.UnderlyingSpan, diag = diagStore)
		{
			if (nrhs > 4)
			{
				InPlaceTranspose(A, lda, n);
				for (int k = 0; k < nrhs; k++)
				{
					// back substitution solve
					T* b = B + k * ldb;
					b[n - 1] /= diag[n - 1];
					for (int i = n - 2; i >= 0; i--)
					{
						int i1 = i + 1;
						T dot = Dot(A + (i1 + i * lda), b + i1, n - i1);
						b[i] = (b[i] - dot) / diag[i];
					}
				}
				InPlaceTranspose(A, lda, n);
			}
			else
			{   // direct access by row, suitable for small number of right hand sides
				for (int k = 0; k < nrhs; k++)
				{
					// back substitution solve
					T* b = B + k * ldb;
					b[n - 1] /= diag[n - 1];
					for (int i = n - 2; i >= 0; i--)
					{
						int i1 = i + 1;
						T dot = T.Zero;
						for (int j = i1; j < n; j++)
							dot += b[j] * A[j * lda + i];
						b[i] = (b[i] - dot) / diag[i];
					}
				}
			}
		}
	}
	#endregion


	#region symmetric eigen
	/// <summary>
	/// Reduce a hermitian <paramref name="matrix"/> to a tridiagonal form stored as <paramref name="diagStore"/> and <paramref name="offDiagStore"/><c>[1..]</c>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="matrix">The input / output <see cref="SpanMatrix{T}"/> to be reduced, replaced by the unary transformation matrix used to transform original matrix to tridiagonal form after return.</param>
	/// <param name="diagStore">The output <see cref="Span{T}"/> to store the diagonal elements of result tridiagonal matrix</param>
	/// <param name="offDiagStore">The output <see cref="Span{T}"/> to store the off diagonal elements of result tridiagonal matrix</param>
	public static void HermitianMatrixToTridiagonal<T>(SpanMatrix<T> matrix, Span<T> diagStore, Span<T> offDiagStore) where T : unmanaged, IFloatingPoint<T>
	{
		if (matrix.IsEmpty)
			return;
		if (matrix.Rows != matrix.Cols)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(matrix));
		int n = matrix.Rows, ld = matrix.LeadDim;
		T two = T.One + T.One;
		fixed (T* A = matrix.UnderlyingSpan, diag = diagStore, offDiag = offDiagStore)
		{
			// reduce to tridiagonal by Householder reflect from the last column
			for (int i = 0; i < n - 2; i++)
			{
				int len = n - i - 1;
				// get Householder reflector and store in A's column i
				// Householder reflector u is generated from A[i+1..,i]
				T* u = A + (i + 1 + i * ld);
				T normSqrU = NormSq(u, len), normU = T.Sqrt(normSqrU);
				normU = T.CopySign(normU, u[0]);
				// get tau and store in A[i, i+1]
				//tex: $\tau = {2}/{\|\vec{u}\|^2}$, $H = I - \tau \vec{u}\vec{u}^*$
				T tau = T.One / (normSqrU + T.Abs(u[0] * normU));
				u[0] += normU;
				A[i + (i + 1) * ld] = tau;
				// get A[i.., i..]
				//tex:$\vec{p} = \tau A_{i+1:,i+1:} \vec{u}$ and store in diag[..i]
				VecMulScaledMat(A + (i + 1 + (i + 1) * ld), ld, tau, u, diag, len, len);
				//tex:$$\vec{p}=\vec{p}-\frac{\tau\vec{u}\cdot\vec{p}}{2}\vec{u}$$
				T k = Dot(u, diag, len) * tau / two;
				AddScaled(diag, u, -k, len);
				//tex:$A_{i+1:,i+1:} = A_{i+1:,i+1:} - \vec{p}\vec{u}^* - \vec{u}\vec{p}^*$
				SymRank2UpdateNeg(A + (i + 1 + (i + 1) * ld), ld, diag, u, len);
				// get beta
				//tex:$$\beta = \mp\|\vec{A}_{i+1:,i}\|$$
				offDiag[i + 1] = -normU;
			}
			// get last off-diagonal
			offDiag[n - 1] = A[n - 1 + (n - 2) * ld];
			// reconstruct unary transformation matrix
			diag[n - 1] = A[n - 1 + (n - 1) * ld];
			diag[n - 2] = A[n - 2 + (n - 2) * ld];
			A[n - 1 + (n - 1) * ld] = A[n - 2 + (n - 2) * ld] = T.One;
			A[n - 2 + (n - 1) * ld] = A[n - 1 + (n - 2) * ld] = T.Zero;
			for (int i = n - 3; i >= 0; i--)
			{
				// get tau and vector u
				int len = n - i - 1;
				T tau = A[i + (i + 1) * ld];
				T* u = A + (i + 1 + i * ld);
				// update Householder reflectors' product stored in A[0..i, 0..i]
				//tex:$Q = Q - \tau \vec{u}_{(i)}\vec{u}_{(i)}^* Q$
				for (int j = i + 1; j < n; j++)
				{
					T dot = Dot(u, A + (i + 1 + j * ld), len);
					AddScaled(A + (i + 1 + j * ld), u, -tau * dot, len);
				}
				// set diag and reset last row and column of A[..i, ..i] for next iteration
				diag[i] = A[i + i * ld];
				A[i + i * ld] = T.One;
				for (int j = i + 1; j < n; j++)
					A[j + i * ld] = A[i + j * ld] = T.Zero;
			}
		}
	}

	/// <summary>
	/// Compute the eigenvalues and eigenvectors of a hermitian tridiagonal matrix represented by <paramref name="diagStore"/> and <paramref name="offDiagStore"/>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="diagStore">The diagonal elements of tridiagonal matrix, can be generated from <see cref="HermitianMatrixToTridiagonal{T}(SpanMatrix{T}, Span{T}, Span{T})"/>, replaced by the eigenvalues after return</param>
	/// <param name="offDiagStore">The off-diagonal elements of tridiagonal matrix, can be generated from <see cref="HermitianMatrixToTridiagonal{T}(SpanMatrix{T}, Span{T}, Span{T})"/>, destroyed after return</param>
	/// <param name="eigenvectors">The input / output <see cref="SpanMatrix{T}"/> to store the multiplication result of <paramref name="eigenvectors"/> and the real eigenvectors of tridiagonal matrix</param>
	/// <returns>Success or not.</returns>
	public static bool HermitianTridiagonalEigensolve<T>(Span<T> diagStore, Span<T> offDiagStore, SpanMatrix<T> eigenvectors) where T : unmanaged, IFloatingPoint<T>
	{
		int n = diagStore.Length, evld = eigenvectors.LeadDim;
		if (offDiagStore.Length != n)
			throw new ArgumentException(Resources.ParameterError.NotSameSize, nameof(offDiagStore));
		if (!eigenvectors.IsEmpty && (eigenvectors.Rows != n || eigenvectors.Cols != n))
			throw new ArgumentException(Resources.ParameterError.NotSameSize, nameof(eigenvectors));

		// constants
		T half = T.One / (T.One + T.One), two = T.One + T.One, four = T.One + T.One + T.One + T.One;
		offDiagStore[0] = T.Zero; // off-diag is advanced by 1
		fixed (T* diag = diagStore, offDiag = offDiagStore, ev = eigenvectors.IsEmpty ? null : eigenvectors.UnderlyingSpan)
		{
			// loop from the eigenvalue at right-bottom to the one at top-left
			for (int k = n - 1; k > 0; k--)
			{
				int iter = 0, i;
			RESTART_EIGVAL:
				for (i = k - 1; i >= 0; i--)
				{
					T d = T.Abs(diag[i]) + T.Abs(diag[i + 1]);
					// look for a single small sub-diagonal element which indicates convergence of one eigenvalue
					if (T.Abs(offDiag[i + 1]) + d == d)
						break;
				}
				if (i == k - 1)
				{   // eigenvalue converged
					continue;
				}
				if (i < 0)
					i = 0;
				// now, A[i..k, i..k] (inclusive k) is tridiagonal in machine precision
				// perform QR with implicit shift
				if (iter++ == 30)
				{   // too many iterations for one eigenvalue, there may be errors
					return false;
				}
				// get eigenvalue shift
				//tex: $s = \text{argmin}_x{|d_k - x|}$ where $x \in \text{eigval}\pmatrix{d_{k-1} & e_{k-1} \\ e_{k-1} & d_k}$
				T s;
				{
					T dSub = diag[k - 1] - diag[k], dAdd = diag[k - 1] + diag[k];
					T sqrt = T.Sqrt(dSub * dSub + four * offDiag[k] * offDiag[k]);
					T s1 = half * (dAdd - sqrt);
					T s2 = half * (dAdd + sqrt);
					if (T.Abs(s1 - diag[k]) <= T.Abs(s2 - diag[k]))
						s = s1;
					else
						s = s2;
				}
				// Householder reflect from the first column
				T c = default;
				for (int j = i; j < k; j++)
				{
					// get Householder reflector matrix [h1, h3; h3, h2]
					//tex:$\gamma=\beta_{j-1}^2+c^2\pm\beta_{j-1}\sqrt{\beta_{j-1}^2+c^2}$, $h_1=-h_2={c^2}/{\gamma}-1$ and $h_3=-{c\left(\beta_{j-1}\pm\sqrt{\beta_{j-1}^2+c^2}\right)}/{\gamma}$
					T b, norm, normSq, γ, h1, h2, h3;
					if (j == i)
					{
						b = diag[i] - s;
						normSq = b * b + offDiag[i + 1] * offDiag[i + 1];
						norm = T.Sqrt(normSq);
						c = offDiag[i + 1];
					}
					else
					{
						b = offDiag[j];
						normSq = c * c + offDiag[j] * offDiag[j];
						norm = T.Sqrt(normSq);
					}
					γ = normSq + T.Abs(b) * norm;
					h3 = -c * (b + T.CopySign(norm, b)) / γ;
					h1 = c * c / γ - T.One; h2 = -h1;
					// update diag and off-diag and c
					//tex:$$\left[\begin{matrix}\beta_{j-1}&\alpha_j&\beta_j&c\\0&\beta_j&\alpha_{j+1}&\beta_{j+1}\\\end{matrix}\right]\gets\left[\begin{matrix}h_1&h_3\\h_3&h_2\\\end{matrix}\right]\left[\begin{matrix}\beta_{j-1}&\alpha_j&\beta_j&0\\c&\beta_j&\alpha_{j+1}&\beta_{j+1}\\\end{matrix}\right]$$
					//$$\left[\begin{matrix}\alpha_j&\beta_j\\\beta_j&\alpha_{j+1}\\\end{matrix}\right]\gets\left[\begin{matrix}\alpha_j&\beta_j\\\beta_j&\alpha_{j+1}\\\end{matrix}\right]\left[\begin{matrix}h_1&h_3\\h_3&h_2\\\end{matrix}\right]$$
					offDiag[j] = h1 * offDiag[j] + h3 * c;
					if (j + 2 < n)
						(offDiag[j + 2], c) = (h2 * offDiag[j + 2], h3 * offDiag[j + 2]);
					(diag[j], diag[j + 1], offDiag[j + 1]) =
						(
						h1 * h1 * diag[j] + h3 * h3 * diag[j + 1] + two * h1 * h3 * offDiag[j + 1],
						h3 * h3 * diag[j] + h2 * h2 * diag[j + 1] + two * h2 * h3 * offDiag[j + 1],
						h3 * (h1 * diag[j] + h2 * diag[j + 1]) + offDiag[j + 1] * (h1 * h2 + h3 * h3)
						);
					// update eigenvectors
					//tex:$$U_{:,j:j+1}\gets U_{:,j:j+1}\left[\begin{matrix}h_1&h_3\\h_3&h_2\\\end{matrix}\right]$$
					if (ev == null)
						continue;
					Householder2x2<T> reflector = new(h2, h3);
					reflector.ColumnUpdate(ev + j * evld, evld, n);
				}
				goto RESTART_EIGVAL;
			}
		}
		return true;
	}
	#endregion


	#region general eigen
	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	private static void CheckGeneralEigen<T>(SpanMatrix<T> matrix, SpanMatrix<T> transformer, Span<T> eigenvalues, Span<T> eigenvaluesImag, out int n, out int lda, out int ldq) where T : unmanaged, IFloatingPoint<T>
	{
		n = lda = ldq = 0;
		if (matrix.IsEmpty)
			return;
		n = matrix.Rows; lda = matrix.LeadDim; ldq = transformer.LeadDim;
		if (matrix.Cols != matrix.Rows)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(matrix));
		if (transformer.Rows != n || transformer.Cols != n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(transformer));
		if (eigenvalues.IsEmpty)
			return;
		if (eigenvalues.Length != n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(eigenvalues));
		if (!NumberType<T>.IsComplex && eigenvaluesImag.IsEmpty)
			throw new ArgumentNullException(nameof(eigenvaluesImag));
		if (!NumberType<T>.IsComplex && eigenvaluesImag.Length != n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(eigenvaluesImag));
	}

	/// <summary>
	/// Reduce the given <paramref name="matrix"/> to a Hessenberg matrix by unary similarity transformation.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="matrix">The input / output <see cref="SpanMatrix{T}"/> to be reduced to Hessenberg form</param>
	/// <param name="transformer">The output <see cref="SpanMatrix{T}"/> to store the unary transformation matrix used to transform <paramref name="matrix"/> to Hessenberg form</param>
	public static void MatrixToHessenberg<T>(SpanMatrix<T> matrix, SpanMatrix<T> transformer) where T : unmanaged, IFloatingPoint<T>
	{
		CheckGeneralEigen(matrix, transformer, default, default, out int n, out int lda, out int ldq);
		if (n == 0)
			return;

		T two = T.One + T.One;
		fixed (T* A = matrix.UnderlyingSpan, Q = transformer.UnderlyingSpan)
		{
			// reduce to Hessenberg form by Householder reflect from the last column
			for (int i = 0; i < n - 2; i++)
			{
				int len = n - i - 1;
				// get Householder reflector and store in A's column i
				// Householder reflector u is generated from A[i+1..,i]
				T* u = ldq == 0 ? Q + 1 : Q + (1 + i * ldq);
				Unsafe.CopyBlockUnaligned(u, A + (i + 1 + i * lda), (uint)(len * sizeof(T)));
				T normSqrU = NormSq(u, len), normU = T.Sqrt(normSqrU);
				normU = T.CopySign(normU, u[0]);
				// get tau and
				//tex: $\tau = {2}/{\|\vec{u}\|^2}$, $H = I - \tau \vec{u}\vec{u}^*$
				T tau = T.One / (normSqrU + T.Abs(u[0] * normU));
				u[0] += normU;
				// get transformed A
				//tex:$A \leftarrow H A H = (A - \tau A\vec{u}\vec{u}^* - \tau \vec{u}\vec{u}^* A + \tau^2 \vec{u}\vec{u}^* A \vec{u}\vec{u}^*)$
				//tex: let $\vec{p} = \tau A \vec{u}$ and store in Q[.., i + 1], and
				//let $\vec{q}^* = \tau \vec{u}^* A$ and store in Q[.., i + 2]
				T* p = ldq == 0 ? Q + n : Q + (i + 1) * ldq;
				T* q = ldq == 0 ? Q + 2 * n : Q + (i + 2) * ldq;
				MatMulScaledVec(A + (i + 1) * lda, lda, tau, u, p, n, len);
				VecMulScaledMat(A + (i + 1), lda, tau, u, q, n, len);
				//tex:$A_{i:,i:} = A_{i:,i:} - \vec{p}\vec{u}^* - \vec{u}\vec{q}^* (I - \tau \vec{u}\vec{u}^*)$
				//tex:let $\vec{q} = \vec{q} - \tau (\vec{u}\cdot\vec{q}) \vec{u}$ then $A = A - \vec{p}\vec{u}^* - \vec{u}\vec{q}^*$
				AddConjugateScaled(q + (i + 1), u, -tau * DotU(q + (i + 1), u, len), len);
				Rank1UpdateNeg(A + (i + 1) * lda, lda, p, u, n, len, true);
				Rank1UpdateNeg(A + (i + 1), lda, u, q, len, n, false);
				// store tau
				u[-1] = tau;
				Unsafe.InitBlockUnaligned(A + (i + 2 + i * lda), 0, (uint)((n - i - 2) * sizeof(T)));
			}
			if (ldq == 0)
				return;
			// reconstruct unary transformation matrix
			Q[n - 1 + (n - 1) * ldq] = Q[n - 2 + (n - 2) * ldq] = T.One;
			Q[n - 2 + (n - 1) * ldq] = Q[n - 1 + (n - 2) * ldq] = T.Zero;
			for (int i = n - 3; i >= 0; i--)
			{
				// get tau and vector u
				int len = n - i - 1;
				T* u = ldq == 0 ? Q + 1 : Q + (1 + i * ldq);
				T tau = u[-1];
				// update Householder reflectors' product stored in Q[..i, ..i]
				//tex:$Q = Q - \tau \vec{u}_{(i)}\vec{u}_{(i)}^* Q$
				for (int j = i + 1; j < n; j++)
				{
					T dot = Dot(u, Q + (i + 1 + j * ldq), len);
					AddScaled(Q + (i + 1 + j * ldq), u, -tau * dot, len);
				}
				// set diag and reset last row and column of Q[..i, ..i] for next iteration
				Q[i + i * ldq] = T.One;
				for (int j = i + 1; j < n; j++)
					Q[j + i * ldq] = Q[i + j * ldq] = T.Zero;
			}
		}
	}

	private static void ToStandardSchurForm<T>(int n, T* A, int lda, T* Q, int ldq, int i) where T : unmanaged, IFloatingPoint<T>
	{
		// constants
		T two = T.One + T.One, half = T.One / two, halfSqrt2 = T.Sqrt(two) * half, four = two + two;
		int k = i + 1;
		// transform to standard Schur form
		//tex:$A_{i:k,:}\gets HA_{i:k,:}$; $A_{:,i:k}\gets A_{:,i:k}H$; $U_{:,i:k}\gets U_{:,i:k}H$,
		//where $h_2,h_3=\frac{\sqrt2}{2}\sqrt{1\pm\frac{\left|b+c\right|}{\sqrt{\left(b+c\right)^2+\left(a-d\right)^2}}}$ for complex;
		//or $h_2,h_3=\sqrt{\frac{2\alpha\left(b+c\right)+\left(a-d\right)\left[a-d\pm\sqrt{4bc+\left(a-d\right)^2}\right]}{2\left[\left(b+c\right)^2+\left(a-d\right)^2\right]}}$, $\alpha = b, c$ for real pair
		T h2, h3;
		T ad = A[i + i * lda] - A[k + k * lda];
		T bc = A[i + k * lda] + A[k + i * lda];
		T b_c = A[i + k * lda] * A[k + i * lda];
		T delta = ad * ad + four * b_c;
		T nrm = bc * bc + ad * ad;
		if (delta >= T.Zero)
		{
			delta = T.Sqrt(delta);
			nrm *= two;
			h2 = (two * A[i + k * lda] * bc + ad * (ad + delta)) / nrm;
			h3 = (two * A[k + i * lda] * bc + ad * (ad - delta)) / nrm;
			h2 = T.Sqrt(h2); h3 = T.Sqrt(h3);
		}
		else
		{
			nrm = T.Sqrt(nrm);
			if (nrm == T.Abs(bc))
				return;
			h2 = halfSqrt2 * T.Sqrt(T.One + bc / nrm);
			h3 = halfSqrt2 * T.Sqrt(T.One - bc / nrm);
			if (ad < T.Zero)
			{
				(h2, h3) = (h3, h2);
			}
		}
		Householder2x2<T> householder = new(h2, h3);
		householder.ColumnUpdate(A + i * lda, lda, k + 1);
		householder.RowUpdate(A + (i + i * lda), lda, n - i);
		householder.ColumnUpdate(Q + i * ldq, ldq, n);
		if (delta >= T.Zero)
			A[k + i * lda] = T.Zero;
	}

	/// <summary>
	/// Compute the Schur factorization of given upper Hessenberg <paramref name="matrix"/>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="matrix">The input / output <see cref="SpanMatrix{T}"/> of Hessenberg form, can be output of <see cref="MatrixToHessenberg{T}(SpanMatrix{T}, SpanMatrix{T})"/>, replaced by its Schur from after return</param>
	/// <param name="transformer">The input / output <see cref="SpanMatrix{T}"/> to be right multiplied by the Schur vectors</param>
	/// <param name="eigenvalues">The output <see cref="Span{T}"/> to store the eigenvalues (or their real parts)</param>
	/// <param name="eigenvaluesImag">The output <see cref="Span{T}"/> to store the eigenvalues' imaginary parts (cannot be empty if <typeparamref name="T"/> is a real type)</param>
	/// <param name="workSpace">The working space with length ≥ 2 * matrix size if <typeparamref name="T"/> is a complex type, not required otherwise</param>
	/// <returns>Success or not.</returns>
	public static bool HessenbergSchurFactorize<T>(SpanMatrix<T> matrix, SpanMatrix<T> transformer, Span<T> eigenvalues, Span<T> eigenvaluesImag = default, Span<T> workSpace = default) where T : unmanaged, IFloatingPoint<T>
	{
		CheckGeneralEigen(matrix, transformer, eigenvalues, eigenvaluesImag, out int n, out int lda, out int ldq);
		if (n == 0)
			return false;
		if (NumberType<T>.IsComplex && workSpace.Length < 2 * n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(workSpace));

		// constants
		T two = T.One + T.One, half = T.One / two, halfSqrt2 = T.Sqrt(two) * half, four = two + two;
		T epsilon = T.ScaleB(T.One, -2 * sizeof(T));
		eigenvaluesImag.Fill(T.Zero);
		eigenvalues[0] = T.NaN;
		fixed (T* A = matrix.UnderlyingSpan, Q = transformer.UnderlyingSpan, wr = eigenvalues, wi = eigenvaluesImag.IsEmpty ? null : eigenvaluesImag, work = workSpace.IsEmpty ? null : workSpace)
		{
			// main loop to compute eigenvalues from bottom to top
			for (int k = n - 1; k > 0; k--)
			{
				int iter = 0, i;
			RESTART_EIGVAL:
				// look for small sub-diagonal to split matrix
				for (i = k; i > 0; i--)
				{
					T d = T.Abs(A[i + i * lda]) + T.Abs(A[i - 1 + (i - 1) * lda]);
					d *= four; // lessen the criteria so that the eigenvalues with multiplicity can converge
					if (T.Abs(A[i + (i - 1) * lda]) + d == d)
					{
						A[i + (i - 1) * lda] = T.Zero;
						break;
					}
				}
				// one eigenvalue converged
				if (i == k)
				{
					wr[k] = A[i + i * lda];
					continue;
				}
				// two eigenvalues converged
				if (!NumberType<T>.IsComplex && i == k - 1 && T.Abs(A[k + (k - 1) * lda]) > epsilon * T.Abs(A[k - 1 + k * lda]))
				{   // prevent 3 real eigenvalues where first one converges while last two does not
					ToStandardSchurForm(n, A, lda, Q, ldq, i);
					T re = A[i + i * lda];
					wr[i] = wr[k] = re;
					re *= re;
					T im = -A[i + k * lda] * A[k + i * lda];
					if (re + im != re)
					{
						wi[i] = T.Sqrt(im);
						wi[k] = -wi[i];
					}
					else
					{
						wr[k] = A[k + k * lda];
						A[k + i * lda] = T.Zero;
					}
					k--;
					continue;
				}
				// no eigenvalue converged
				if (i < 0)
					i = 0;
				if (iter++ == 30)
				{   // too many iterations, there may be errors
					return false;
				}
				// continue implicit QR iteration for A[i..k, i..k] (inclusive)
				for (int j = i; j < k; j++)
				{
					Householder3<T> householder = default;
					if (j == i)
					{
						//tex:$$\vec{v}\gets \begin{pmatrix}{\left[\left(a_{k,k}-a_{i,i}\right)\left(a_{k-1,k-1}-a_{i,i}\right)-a_{k-1,k}a_{k,k-1}\right]}/{a_{i+1,i}}+a_{i,i+1} \\ a_{i,i}+a_{i+1,i+1}-\left(a_{k-1,k-1}+a_{k,k}\right) \\ a_{i+2,i+1} \end{pmatrix}$$
						T akk = A[k + k * lda], aii = A[i + i * lda],
						  ak1k1 = A[k - 1 + (k - 1) * lda], ai1i1 = A[i + 1 + (i + 1) * lda],
						  ak1k = A[k - 1 + k * lda], akk1 = A[k + (k - 1) * lda],
						  ai1i = A[i + 1 + i * lda], aii1 = A[i + (i + 1) * lda];
						householder = new
						(
							((akk - aii) * (ak1k1 - aii) - ak1k * akk1) + aii1 * ai1i,
							(aii + ai1i1 - (ak1k1 + akk)) * ai1i,
							A[i + 2 + (i + 1) * lda] * ai1i
						); // do not use division to prevent overflow when near convergence
					}
					else
					{
						//tex:$\vec{v}\gets{\vec{a}}_{j:j+2,j-1}$
						householder = new(A[j + (j - 1) * lda], A[j + 1 + (j - 1) * lda], A[j + 2 + (j - 1) * lda]);
					}
					//tex:$H\gets I - 2\vec{v}\vec{v}^* / \|v\|^2$
					//tex:$A_{j:j+2,:}\gets HA_{j:j+2,:}$; $A_{:,j:j+2}\gets A_{:,j:j+2}H$; $U_{:,j:j+2}\gets U_{:,j:j+2}H$
					int colFree = j == i ? i : j - 1;
					if (j == k - 1)
					{
						Householder2<T> householder2 = new(A[j + (j - 1) * lda], A[j + 1 + (j - 1) * lda]);
						var reflector = householder2.ToReflectionMatrix();
						reflector.ColumnUpdate(A + (j * lda), lda, Math.Min(j + 4, n), work);
						reflector.RowUpdate(A + (j + colFree * lda), lda, n - colFree);
						reflector.ColumnUpdate(Q + (j * ldq), ldq, n, work);
						A[j + 1 + (j - 1) * lda] = T.Zero;
					}
					else
					{
						var reflector = householder.ToReflectionMatrix();
						reflector.ColumnUpdate(A + (j * lda), lda, Math.Min(j + 4, n), work);
						reflector.RowUpdate(A + (j + colFree * lda), lda, n - colFree);
						reflector.ColumnUpdate(Q + (j * ldq), ldq, n, work);
						if (j != i)
							A[j + 1 + (j - 1) * lda] = A[j + 2 + (j - 1) * lda] = T.Zero;
					}
				}
				// restart iteration for this eigenvalue
				goto RESTART_EIGVAL;
			}
			// get first eigenvalue and return
			if (T.IsNaN(wr[0]))
				wr[0] = A[0];
		}
		return true;
	}

	/// <summary>
	/// Sort the Schur factorization result (possibly generated from <see cref="HessenbergSchurFactorize{T}(SpanMatrix{T}, SpanMatrix{T}, Span{T}, Span{T}, Span{T})"/>) by the given <paramref name="keys"/> corresponding to the <paramref name="eigenvalues"/>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <typeparam name="TKey">The data type of keys</typeparam>
	/// <param name="keys">The <see cref="Span{T}"/> of keys as the eigenvalue sort basis, must be the same for eigenvalues with same value or conjugate, replaced by stably ascending sorted result</param>
	/// <param name="matrix">The input / output Schur form <see cref="SpanMatrix{T}"/> to be sorted by <paramref name="keys"/></param>
	/// <param name="transformer">The input / output Schur vector <see cref="SpanMatrix{T}"/> to be sorted by <paramref name="keys"/></param>
	/// <param name="eigenvalues">The input / output eigenvalues to be sorted by <paramref name="keys"/></param>
	/// <param name="eigenvaluesImag">The input / output eigenvalues' imaginary parts to be sorted by <paramref name="keys"/></param>
	/// <param name="workSpace">The additional working space, length ≥ 3 * matrix size when <typeparamref name="T"/> is complex, not required otherwise</param>
	public static void ReorderSchurForm<T, TKey>(Span<TKey> keys, SpanMatrix<T> matrix, SpanMatrix<T> transformer, Span<T> eigenvalues, Span<T> eigenvaluesImag = default, Span<T> workSpace = default) where T : unmanaged, IFloatingPoint<T> where TKey : IComparisonOperators<TKey, TKey>
	{
		CheckGeneralEigen(matrix, transformer, eigenvalues, eigenvaluesImag, out int n, out int lda, out int ldq);
		if (n == 0)
			return;
		if (keys.Length != n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(keys));
		if (NumberType<T>.IsComplex && workSpace.Length < 3 * n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(workSpace));

		// work spaces
		Span<T> vecX = stackalloc T[4], diag = stackalloc T[4];
		fixed (T* A = matrix.UnderlyingSpan, Q = transformer.UnderlyingSpan, wr = eigenvalues, wi = eigenvaluesImag.IsEmpty ? null : eigenvaluesImag, work = workSpace.IsEmpty ? null : workSpace)
		{
			// bubble sort outer loop
			for (int k = 1; k < n; k++)
			{
				// bubble sort inner loop
				for (int i = n - 1; i >= k;)
				{
					// get size
					int q = A[i + (i - 1) * lda] != T.Zero ? 2 : 1;
					int p = i >= q + 1 && A[i - q + (i - q - 1) * lda] != T.Zero ? 2 : 1;
					if (keys[i] >= keys[i - q])
					{
						i -= q;
						continue;
					}
					// 4 cases
					if (p == 1 && q == 1)
					{
						int j = i - 1;
						// swap simple ones
						(keys[i], keys[j]) = (keys[j], keys[i]);
						(wr[i], wr[j]) = (wr[j], wr[i]);
						if (wi != null)
							(wi[i], wi[j]) = (wi[j], wi[i]);
						// set matrix T
						T alpha1 = A[j + j * lda];
						T alpha2 = A[i + i * lda];
						T t = A[j + i * lda];
						// solve X
						T x = t / (alpha1 - alpha2);
						// QR factorize
						var reflector = new Householder2<T>(-x, T.One).ToReflectionMatrix();
						// swap blocks
						reflector.ColumnUpdate(A + j * lda, lda, i + 1, work);
						reflector.RowUpdate(A + (j + j * lda), lda, n - j);
						reflector.ColumnUpdate(Q + j * lda, ldq, n, work);
						A[i + j * lda] = T.Zero;
					}
					else if (p == 2 && q == 1)
					{
						int jj = i - 2, j = i - 1;
						// swap simple ones
						(keys[i], keys[jj]) = (keys[jj], keys[i]);
						(wr[i], wr[jj]) = (wr[jj], wr[i]);
						if (wi != null)
							(wi[jj], wi[j], wi[i]) = (wi[i], wi[jj], wi[j]);
						// set matrix T
						T alpha1 = A[jj + jj * lda], delta1 = A[j + j * lda], beta1 = A[jj + j * lda], gamma1 = A[j + jj * lda];
						T t1 = A[jj + i * lda], t3 = A[j + i * lda];
						T alpha2 = A[i + i * lda];
						// solve X
						//tex:$$x_1\gets \frac{t_1 \left(\alpha _2-\delta _1\right)+\beta _1 t_3}{\left(\alpha _1-\alpha _2\right) \left(\alpha _2-\delta _1\right)+\beta _1 \gamma _1},x_2\gets \frac{\left(\alpha _2-\alpha _1\right) t_3+\gamma _1 t_1}{\left(\alpha _1-\alpha _2\right) \left(\alpha _2-\delta _1\right)+\beta _1 \gamma _1}$$
						T denom = T.One / ((alpha1 - alpha2) * (alpha2 - delta1) + beta1 * gamma1);
						T x1 = ((alpha2 - delta1) * t1 + beta1 * t3) * denom;
						T x2 = ((alpha2 - alpha1) * t3 + gamma1 * t1) * denom;
						// QR factorize
						var reflector = new Householder3<T>(-x1, -x2, T.One).ToReflectionMatrix();
						// swap blocks
						reflector.ColumnUpdate(A + jj * lda, lda, i + 1, work);
						reflector.RowUpdate(A + (jj + jj * lda), lda, n - jj);
						reflector.ColumnUpdate(Q + jj * lda, ldq, n, work);
						A[j + jj * lda] = A[i + jj * lda] = T.Zero;
					}
					else if (p == 1 && q == 2)
					{
						int j = i - 2, ii = i - 1;
						// swap simple ones
						(keys[i], keys[j]) = (keys[j], keys[i]);
						(wr[i], wr[j]) = (wr[j], wr[i]);
						if (wi != null)
							(wi[j], wi[ii], wi[i]) = (wi[ii], wi[i], wi[j]);
						// set matrix T
						T alpha1 = A[j + j * lda];
						T t3 = A[j + ii * lda], t4 = A[j + i * lda];
						T alpha2 = A[ii + ii * lda], delta2 = A[i + i * lda], beta2 = A[ii + i * lda], gamma2 = A[i + ii * lda];
						// solve X
						//tex:$$x_1 \gets \frac{t_3 \left(\delta _2-\alpha _1\right)-\gamma _2 t_4}{\beta _2 \gamma _2-\left(\alpha _1-\alpha _2\right) \left(\alpha _1-\delta _2\right)}, x_2 \gets \frac{\left(\alpha _1-\alpha _2\right) t_4+\beta _2 t_3}{\left(\alpha _1-\alpha _2\right) \left(\alpha _1-\delta _2\right)-\beta _2 \gamma _2}$$
						T denom = T.One / ((alpha1 - alpha2) * (alpha1 - delta2) - beta2 * gamma2);
						T x1 = (gamma2 * t4 + t3 * (alpha1 - delta2)) * denom;
						T x2 = (beta2 * t3 + t4 * (alpha1 - alpha2)) * denom;
						// QR factorize
						Orthogonal3x3<T> reflector = new(-x1, T.One, T.Zero, -x2, T.Zero, T.One, default, default, default);
						var matReflect = new SpanMatrix<T>(new Span<T>((T*)&reflector, 9), 3);
						QrFactorize(matReflect[..2], diag);
						QrGenerateQ(0, 3, matReflect[..2], diag);
						// swap blocks
						reflector.ColumnUpdate(A + j * lda, lda, i + 1, work);
						reflector.RowUpdate(A + (j + j * lda), lda, n - j);
						reflector.ColumnUpdate(Q + j * lda, ldq, n, work);
						A[i + j * lda] = A[i + ii * lda] = T.Zero;
					}
					else //if (p == 2 && q == 2)
					{
						int jj = i - 3, j = i - 2, ii = i - 1;
						// swap simple ones
						(keys[ii], keys[i], keys[jj], keys[j]) = (keys[jj], keys[j], keys[ii], keys[i]);
						(wr[ii], wr[i], wr[jj], wr[j]) = (wr[jj], wr[j], wr[ii], wr[i]);
						if (wi != null)
							(wi[ii], wi[i], wi[jj], wi[j]) = (wi[jj], wi[j], wi[ii], wi[i]);
						// set matrix T
						//tex:$$T \gets \begin{pmatrix} \alpha _1-\alpha _2 & -\gamma _2 & \beta _1 & 0 \\ -\beta _2 & \alpha _1-\delta _2 & 0 & \beta _1 \\ \gamma _1 & 0 & \delta _1-\alpha _2 & -\gamma _2 \\ 0 & \gamma _1 & -\beta _2 & \delta _1-\delta _2 \end{pmatrix}$$
						T alpha1 = A[jj + jj * lda], delta1 = A[j + j * lda], beta1 = A[jj + j * lda], gamma1 = A[j + jj * lda];
						T alpha2 = A[ii + ii * lda], delta2 = A[i + i * lda], beta2 = A[ii + i * lda], gamma2 = A[i + ii * lda];
						Orthogonal4x4<T> reflector = default;
						T* matT = (T*)&reflector;
						matT[0] = alpha1 - alpha2; matT[1] = -beta2; matT[2] = gamma1;
						matT[4] = -gamma2; matT[5] = alpha1 - delta2; matT[7] = gamma1;
						matT[8] = beta1; matT[10] = delta1 - alpha2; matT[11] = -beta2;
						matT[13] = beta1; matT[14] = -gamma2; matT[15] = delta1 - delta2;
						vecX[0] = A[jj + ii * lda]; vecX[1] = A[jj + i * lda];
						vecX[2] = A[j + ii * lda]; vecX[3] = A[j + i * lda];
						// solve X
						var matReflect = new SpanMatrix<T>(new Span<T>(matT, 16), 4);
						QrFactorize(matReflect, diag);
						QrQtMultiply(matReflect, new(vecX, 4));
						QrLinearSolve(matReflect, diag, new(vecX, 4));
						// QR factorize
						reflector = default;
						matT[0] = -vecX[0]; matT[1] = -vecX[2]; matT[4] = -vecX[1]; matT[5] = -vecX[3];
						matT[2] = T.One; matT[7] = T.One;
						QrFactorize(matReflect[..2], diag);
						QrGenerateQ(0, 4, matReflect[..2], diag);
						// swap blocks
						reflector.ColumnUpdate(A + jj * lda, lda, i + 1, work);
						reflector.RowUpdate(A + (jj + jj * lda), lda, n - jj);
						reflector.ColumnUpdate(Q + jj * lda, ldq, n, work);
						A[ii + jj * lda] = A[ii + j * lda] = A[i + jj * lda] = A[i + j * lda] = T.Zero;
					}
					i -= p;
				}
				// to standard Schur form if necessary
				if (A[k + (k - 1) * lda] != T.Zero)
					ToStandardSchurForm(n, A, lda, Q, ldq, k - 1);
			}
		}
	}

	/// <summary>
	/// Compute the right eigenvectors of given Schur form <paramref name="matrix"/>.
	/// </summary>
	/// <typeparam name="T">The floating point data type</typeparam>
	/// <param name="matrix">The input Schur form <see cref="SpanMatrix{T}"/> to obtain eigenvectors</param>
	/// <param name="transformer">The input Schur vector <see cref="SpanMatrix{T}"/> to left multiply eigenvectors of <paramref name="matrix"/></param>
	/// <param name="eigenvectors">The output <see cref="SpanMatrix{T}"/> to store the multiplication result of <paramref name="transformer"/> and eigenvectors of <paramref name="matrix"/></param>
	/// <param name="workSpace">The working space with length ≥ 9 * matrix size</param>
	/// <param name="eigenvalues">The input eigenvalues</param>
	/// <param name="eigenvaluesImag">The input eigenvalues' imaginary parts if <typeparamref name="T"/> is a real type</param>
	public static void SchurFormEigensolve<T>(SpanMatrix<T> matrix, SpanMatrix<T> transformer, SpanMatrix<T> eigenvectors, Span<T> workSpace, Span<T> eigenvalues, Span<T> eigenvaluesImag = default) where T : unmanaged, IFloatingPoint<T>
	{
		CheckGeneralEigen(matrix, transformer, eigenvalues, eigenvaluesImag, out int n, out int lda, out int ldq);
		if (n == 0)
			return;
		int ldv = eigenvectors.LeadDim;
		if (eigenvectors.Rows != n || eigenvectors.Cols != n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(eigenvectors));
		if (workSpace.Length < 9 * n)
			throw new ArgumentException(Resources.ParameterError.WrongSize, nameof(workSpace));

		// constant
		T criteriaAmplifier = T.ScaleB(T.One, sizeof(T) * 3 / 2), sqrtCriteriaAmplifier = T.Sqrt(criteriaAmplifier);
		fixed (T* A = matrix.UnderlyingSpan, Q = transformer.UnderlyingSpan, V = eigenvectors.UnderlyingSpan,
				  wr = eigenvalues, wi = eigenvaluesImag.IsEmpty ? null : eigenvaluesImag, work = workSpace)
		{   // store some diagonals
			T* diag_m1 = work, diag_0 = work + n, diag_p1 = work + 2 * n;
			diag_m1[0] = diag_p1[0] = diag_0[0] = T.Zero;
			for (int i = 0; i < n - 1; i++)
			{
				int ip = i + 1;
				diag_m1[ip] = -A[ip + i * lda];
				diag_p1[ip] = A[i + ip * lda];
				diag_0[ip] = A[i + i * lda];
			}
			// main loop
			for (int k = 0; k < n;)
			{
				T* alpha = work + 3 * n, alphaIm = work + 4 * n;
				T* beta = work + 5 * n, betaIm = work + 6 * n;
				T* temp = work + 7 * n, tempIm = work + 8 * n;
				T* transformTemp = null;
				if (NumberType<T>.IsComplex)
				{
					alpha = work + 3 * n; beta = work + 4 * n; temp = work + 5 * n;
					transformTemp = work + 6 * n;
				}
				// find last eigenvalue equals or conjugates
				int l;
				T λ = wr[k];
				bool noComplex = wi == null || wi[k] == T.Zero;
				if (noComplex)
				{
					// lessen the criteria
					T λAbs = sqrtCriteriaAmplifier * T.Abs(λ);
					for (l = k; l < n; l++)
					{
						T diff = T.Abs(wr[l] - λ);
						if ((wi != null && wi[l] != T.Zero) || diff + λAbs != λAbs)
							break;
					}
					alphaIm = betaIm = tempIm = null;
					λ = λAbs;
				}
				else
				{
					// lessen the criteria
					T λAbsSq = criteriaAmplifier * (λ * λ + wi[k] * wi[k]);
					for (l = k; l < n; l++)
					{
						T diff = (wr[l] - λ) * (wr[l] - λ) + (T.Abs(wi[l]) - wi[k]) * (T.Abs(wi[l]) - wi[k]);
						if (wi[l] == T.Zero || diff + λAbsSq != λAbsSq)
							break;
					}
					λ = T.Sqrt(λAbsSq);
				}
				// get columns whose eigenvectors are not 0 and copy to V[..k, k..l]
				//tex:$\mathcal{I}\gets\left\{i\middle|{\vec{a}}_{k:l,i}={\vec{e}}_{i-k+1}\lambda_k,k\le i < l\right\}$
				int zeroColCount = 0;
				for (int i = k; i < l; i++)
				{
					if (noComplex && !AllZeroComparedTo(A + (k + i * lda), i - k, λ))
						continue;
					if (!noComplex && ((i - k) % 2 == 1 || !AllZeroComparedTo(A + (k + i * lda), i - k, λ) || !AllZeroComparedTo(A + (k + (i + 1) * lda), i - k, λ)))
						continue;
					if (k != 0)
					{
						if (noComplex)
							Unsafe.CopyBlockUnaligned(V + ((zeroColCount + k) * ldv), A + (i * lda), (uint)(k * sizeof(T)));
						else
							Unsafe.CopyBlockUnaligned(V + ((zeroColCount + k) * ldv), A + ((i + 1) * lda), (uint)((k + 1) * sizeof(T)));
					}
					zeroColCount++;
				}
				// real end row number for complex
				int kk = noComplex ? k : k + 1;
				// first eigenvectors shortcut
				if (k == 0)
				{
					if (!noComplex)
						zeroColCount *= 2;
					for (int i = 0; i < zeroColCount; i++)
					{
						Unsafe.CopyBlockUnaligned(V + (i * ldv), Q + (i * ldq), (uint)(n * sizeof(T)));
					}
					for (int i = zeroColCount; i < l; i++)
					{
						Unsafe.InitBlockUnaligned(V + (i * ldv), 0, (uint)(n * sizeof(T)));
					}
					if (!noComplex)
					{
						//tex:$$\vec{v} = \left[ \pm \frac{\sqrt{b}}{\sqrt{c-b}},\frac{1}{\sqrt{\left| {b}/{c}\right| +1}} \right]$$
						for (int i = 0; i < zeroColCount; i += 2)
						{
							T b = A[i + (i + 1) * lda], c = A[i + 1 + i * lda];
							T im = T.Sqrt(T.Abs(b / (c - b))), re = T.One / T.Sqrt(T.Abs(b / c) + T.One);
							// store real and imaginary parts in two columns
							Scale(V + i * ldv, im, n);
							Scale(V + (i + 1) * ldv, re, n);
						}
					}
					k = l; continue;
				}
				// get work vector alpha and beta for row reduction
				//tex:$\vec{\beta}\gets-{{\rm \text{diag}}_{-1}{A_{0:k-1,0:k-1}}}/{\left(\text{diag}{A_{0:k-2,0:k-2}}-\lambda_k\right)}$
				//$\vec{\alpha}\gets{1}/{\left(\text{diag}{A_{1:k-1,1:k-1}}+\vec{\beta}\odot{\rm \text{diag}}_1{A_{0:k-1,0:k-1}}-\lambda_k\right)}$
				//$\vec{\beta}\gets\vec{\alpha}\odot\vec{\beta}$
				PointWiseDivideAddScalar(diag_m1, diag_0, kk, -wr[k], wi == null ? default : -wi[k], beta, betaIm);
				PointWiseMultiplyAddScalar(beta, betaIm, diag_p1, diag_0 + 1, kk, -wr[k], wi == null ? default : -wi[k], alpha, alphaIm);
				PointWiseInv(alpha, alphaIm, kk, alpha, alphaIm);
				PointWiseMultiply(alpha, alphaIm, beta, betaIm, kk, beta, betaIm);
				// move beta forward for future computation
				for (int i = 0; i < kk - 1; i++)
					beta[i] = beta[i + 1];
				if (betaIm != null)
					for (int i = 0; i < k; i++)
						betaIm[i] = betaIm[i + 1];
				beta[kk - 1] = T.Zero;
				if (betaIm != null)
					betaIm[k] = T.Zero;
				// row reduce V[..k, k..l]
				//tex:$$V_{1:k-1,\mathcal{I}}\gets \text{diag}{\vec{\beta}}\cdot V_{0:k-2,\mathcal{I}}+\text{diag}{\vec{\alpha}}\cdot V_{1:k-1,\mathcal{I}}$$
				//For j = k - 2, ..., 1 Do
				//$$V_{1:j,\mathcal{I}}\gets V_{1:j,\mathcal{I}}-\left({\vec{\beta}}_{1:j}\odot{\vec{a}}_{0:j-1,j+1}+{\vec{\alpha}}_{1:j}\odot{\vec{a}}_{1:j,j+1}\right)\otimes{\vec{v}}_{k-1,\mathcal{I}}$$
				for (int i = 0; i < zeroColCount; i++)
				{
					T* Vre = noComplex ? V + (k + i) * ldv : V + (k + i * 2) * ldv;
					T* Vim = noComplex ? null : V + (k + i * 2 + 1) * ldv;
					PointWiseMultiplyAdd(beta, betaIm, Vre, alpha + 1, alphaIm + 1, Vre + 1, kk - 1, temp + 1, tempIm + 1);
					temp[0] = alpha[0] * Vre[0];
					if (!noComplex)
						tempIm[0] = alphaIm[0] * Vre[0];
					Unsafe.CopyBlockUnaligned(Vre, temp, (uint)(kk * sizeof(T)));
					if (!noComplex)
						Unsafe.CopyBlockUnaligned(Vim, tempIm, (uint)(kk * sizeof(T)));
				}
				for (int j = kk - 1; j > 0; j--)
				{
					PointWiseMultiplyAdd(beta, betaIm, A + j * lda, alpha + 1, alphaIm + 1, A + (1 + j * lda), j - 1, temp + 1, tempIm + 1);
					temp[0] = alpha[0] * A[j * lda];
					if (!noComplex)
						tempIm[0] = alphaIm[0] * A[j * lda];
					for (int i = 0; i < zeroColCount; i++)
					{
						T* Vre = noComplex ? V + (k + i) * ldv : V + (k + i * 2) * ldv;
						T* Vim = noComplex ? null : V + (k + i * 2 + 1) * ldv;
						PointWiseAddScaled(Vre, Vim, temp, tempIm, j, -Vre[j], noComplex ? default : -Vim[j], Vre, Vim);
					}
				}
				// from eigenvectors
				//tex:$V_{k:n,\mathcal{I}}\gets-\left[{\vec{e}}_1,\ldots,{\vec{e}}_{\left|\mathcal{I}\right|}\right]$; 
				//$V_{:,\mathcal{I}}\gets\text{Orthogonalize}\left(V_{:,\mathcal{I}}\right)$; 
				//$V_{:,\mathcal{I}}\gets U\cdot V_{:,\mathcal{I}}$
				for (int i = 0; i < zeroColCount; i++)
				{
					if (noComplex)
					{
						Unsafe.InitBlockUnaligned(V + (k + (k + i) * ldv), 0, (uint)((n - k) * sizeof(T)));
						V[k + i + (k + i) * ldv] = -T.One;
					}
					else
					{
						Unsafe.InitBlockUnaligned(V + (kk + (k + i * 2) * ldv), 0, (uint)((n - kk) * sizeof(T)));
						Unsafe.InitBlockUnaligned(V + (kk + (k + i * 2 + 1) * ldv), 0, (uint)((n - kk) * sizeof(T)));
						V[kk + i * 2 + (k + i * 2) * ldv] = -T.One;
					}
				}
				Orthogonalize(!noComplex, kk + zeroColCount, zeroColCount, V + k * ldv, ldv);
				if (!noComplex)
					zeroColCount *= 2;
				for (int i = 0; i < zeroColCount; i++)
				{
					MatMulVec(Q, ldq, V + (k + i) * ldv, temp, n, k + i + 1);
					Unsafe.CopyBlockUnaligned(V + (k + i) * ldv, temp, (uint)(n * sizeof(T)));
				}
				for (int i = zeroColCount + k; i < l; i++)
				{
					Unsafe.InitBlockUnaligned(V + i * ldv, 0, (uint)(n * sizeof(T)));
				}
				// continue loop
				k = l;
			}
		}
	}
	#endregion
}
