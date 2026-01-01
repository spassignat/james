// Fonctions de dessin des figures géométriques
const FigureDrawer = {
	drawSquare(ctx, centerX, centerY, side) {
		const size = side * 15;
		const x = centerX - size / 2;
		const y = centerY - size / 2;

		ctx.strokeStyle = '#3498db';
		ctx.lineWidth = 2;
		ctx.strokeRect(x, y, size, size);

		ctx.fillStyle = '#333';
		ctx.font = '14px Arial';
		ctx.textAlign = 'center';
		ctx.fillText(`${side} cm`, centerX, y - 10);
		ctx.textAlign = 'right';
		ctx.fillText(`${side} cm`, x - 10, centerY);
	},

	drawRectangle(ctx, centerX, centerY, length, width) {
		const scaledLength = length * 10;
		const scaledWidth = width * 10;
		const x = centerX - scaledLength / 2;
		const y = centerY - scaledWidth / 2;

		ctx.strokeStyle = '#3498db';
		ctx.lineWidth = 2;
		ctx.strokeRect(x, y, scaledLength, scaledWidth);

		ctx.fillStyle = '#333';
		ctx.font = '14px Arial';
		ctx.textAlign = 'center';
		ctx.fillText(`${length} cm`, centerX, y - 10);
		ctx.textAlign = 'right';
		ctx.fillText(`${width} cm`, x - 10, centerY);
	},

	drawCircle(ctx, centerX, centerY, radius) {
		const scaledRadius = radius * 10;

		ctx.strokeStyle = '#3498db';
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.arc(centerX, centerY, scaledRadius, 0, 2 * Math.PI);
		ctx.stroke();

		ctx.strokeStyle = '#e74c3c';
		ctx.setLineDash([5, 3]);
		ctx.beginPath();
		ctx.moveTo(centerX, centerY);
		ctx.lineTo(centerX + scaledRadius, centerY);
		ctx.stroke();
		ctx.setLineDash([]);

		ctx.fillStyle = '#333';
		ctx.font = '14px Arial';
		ctx.textAlign = 'center';
		ctx.fillText(`r = ${radius} cm`, centerX + scaledRadius / 2, centerY + 20);
	},

	drawTriangle(ctx, centerX, centerY, base, height) {
		const scaledBase = base * 10;
		const scaledHeight = height * 10;
		const x = centerX - scaledBase / 2;
		const y = centerY + scaledHeight / 2;

		ctx.strokeStyle = '#3498db';
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.moveTo(x, y);
		ctx.lineTo(x + scaledBase, y);
		ctx.lineTo(x + scaledBase / 2, y - scaledHeight);
		ctx.closePath();
		ctx.stroke();

		ctx.fillStyle = '#333';
		ctx.font = '14px Arial';
		ctx.textAlign = 'center';
		ctx.fillText(`${base} cm`, centerX, y + 20);
		ctx.textAlign = 'left';
		ctx.fillText(`${height} cm`, x + scaledBase + 10, centerY);
	},

	drawCube(ctx, centerX, centerY, side) {
		const size = side * 12;
		const perspective = 0.3;

		ctx.strokeStyle = '#3498db';
		ctx.lineWidth = 2;
		ctx.strokeRect(centerX - size/2, centerY - size/2, size, size);

		ctx.beginPath();
		ctx.moveTo(centerX - size/2, centerY - size/2);
		ctx.lineTo(centerX - size/2 - size*perspective, centerY - size/2 - size*perspective);
		ctx.lineTo(centerX + size/2 - size*perspective, centerY - size/2 - size*perspective);
		ctx.lineTo(centerX + size/2, centerY - size/2);
		ctx.stroke();

		ctx.beginPath();
		ctx.moveTo(centerX + size/2, centerY - size/2);
		ctx.lineTo(centerX + size/2 - size*perspective, centerY - size/2 - size*perspective);
		ctx.lineTo(centerX + size/2 - size*perspective, centerY + size/2 - size*perspective);
		ctx.lineTo(centerX + size/2, centerY + size/2);
		ctx.stroke();

		ctx.fillStyle = '#333';
		ctx.font = '14px Arial';
		ctx.textAlign = 'center';
		ctx.fillText(`${side} cm`, centerX, centerY - size/2 - 10);
	},

	drawRectangularPrism(ctx, centerX, centerY, length, width, height) {
		const scaledLength = length * 8;
		const scaledWidth = width * 8;
		const scaledHeight = height * 8;
		const perspective = 0.3;

		ctx.strokeStyle = '#3498db';
		ctx.lineWidth = 2;
		ctx.strokeRect(centerX - scaledLength/2, centerY - scaledHeight/2, scaledLength, scaledHeight);

		ctx.beginPath();
		ctx.moveTo(centerX - scaledLength/2, centerY - scaledHeight/2);
		ctx.lineTo(centerX - scaledLength/2 - scaledWidth*perspective, centerY - scaledHeight/2 - scaledWidth*perspective);
		ctx.lineTo(centerX + scaledLength/2 - scaledWidth*perspective, centerY - scaledHeight/2 - scaledWidth*perspective);
		ctx.lineTo(centerX + scaledLength/2, centerY - scaledHeight/2);
		ctx.stroke();

		ctx.beginPath();
		ctx.moveTo(centerX + scaledLength/2, centerY - scaledHeight/2);
		ctx.lineTo(centerX + scaledLength/2 - scaledWidth*perspective, centerY - scaledHeight/2 - scaledWidth*perspective);
		ctx.lineTo(centerX + scaledLength/2 - scaledWidth*perspective, centerY + scaledHeight/2 - scaledWidth*perspective);
		ctx.lineTo(centerX + scaledLength/2, centerY + scaledHeight/2);
		ctx.stroke();

		ctx.fillStyle = '#333';
		ctx.font = '14px Arial';
		ctx.textAlign = 'center';
		ctx.fillText(`${length} cm`, centerX, centerY - scaledHeight/2 - 10);
		ctx.textAlign = 'left';
		ctx.fillText(`${height} cm`, centerX + scaledLength/2 + 10, centerY);
		ctx.textAlign = 'right';
		ctx.fillText(`${width} cm`, centerX - scaledLength/2 - 10, centerY - scaledHeight/4);
	},

	drawSphere(ctx, centerX, centerY, radius) {
		const scaledRadius = radius * 10;

		ctx.strokeStyle = '#3498db';
		ctx.lineWidth = 2;
		ctx.beginPath();
		ctx.arc(centerX, centerY, scaledRadius, 0, 2 * Math.PI);
		ctx.stroke();

		ctx.fillStyle = '#333';
		ctx.font = '14px Arial';
		ctx.textAlign = 'center';
		ctx.fillText(`r = ${radius} cm`, centerX + scaledRadius / 2, centerY + 20);
	},

	drawCylinder(ctx, centerX, centerY, radius, height) {
		const scaledRadius = radius * 15;
		const scaledHeight = height * 8;

		// Corps du cylindre
		ctx.strokeStyle = '#3498db';
		ctx.lineWidth = 2;

		// Ellipse du haut
		ctx.beginPath();
		ctx.ellipse(centerX, centerY - scaledHeight/2, scaledRadius, scaledRadius/3, 0, 0, 2 * Math.PI);
		ctx.stroke();

		// Ellipse du bas
		ctx.beginPath();
		ctx.ellipse(centerX, centerY + scaledHeight/2, scaledRadius, scaledRadius/3, 0, 0, 2 * Math.PI);
		ctx.stroke();

		// Côtés
		ctx.beginPath();
		ctx.moveTo(centerX - scaledRadius, centerY - scaledHeight/2);
		ctx.lineTo(centerX - scaledRadius, centerY + scaledHeight/2);
		ctx.stroke();

		ctx.beginPath();
		ctx.moveTo(centerX + scaledRadius, centerY - scaledHeight/2);
		ctx.lineTo(centerX + scaledRadius, centerY + scaledHeight/2);
		ctx.stroke();

		ctx.fillStyle = '#333';
		ctx.font = '14px Arial';
		ctx.textAlign = 'center';
		ctx.fillText(`r = ${radius} m`, centerX + scaledRadius + 20, centerY);
		ctx.fillText(`h = ${height} m`, centerX, centerY - scaledHeight/2 - 20);
	}
};