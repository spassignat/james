import {defineConfig} from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'path';

const pathSrc = path.resolve(__dirname, 'src');
export default defineConfig({
	plugins: [vue(),
	],
	resolve: {
		alias: {
			'~/': `${pathSrc}/`,
			'@': path.resolve(__dirname, `${pathSrc}/`)
		},
	},
	root: "./src",
	build: {
		outDir: "./dist/"
	},
	css: {
		preprocessorOptions: {
			scss: {
				additionalData: `@use "~/styles/element/index.scss" as *;`,
			},
		},
	}, transformAssetUrlsOptions: {
		base: null,
		includeAbsolute: false,
	},
})
