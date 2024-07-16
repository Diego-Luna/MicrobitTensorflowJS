#!/bin/sh

LF=$(printf '\\\012_')
LF=${LF%_}
EXTENSION_ID=microbitTensorflowJS
COLLABORATOR=MoonMakers
EXTENSION_NAME='micro:bit with TensorflowJS'
EXTENSION_DESCRIPTION="Extensión de MoonMakers para usar TensorflowJS en BlockCode"

npm install

mkdir -p node_modules/scratch-vm/src/extensions/scratch3_${EXTENSION_ID}
cp ${EXTENSION_ID}/scratch-vm/src/extensions/scratch3_${EXTENSION_ID}/index.js node_modules/scratch-vm/src/extensions/scratch3_${EXTENSION_ID}/
mv node_modules/scratch-vm/src/extension-support/extension-manager.js node_modules/scratch-vm/src/extension-support/extension-manager.js_orig
sed -e "s|class ExtensionManager {$|builtinExtensions['${EXTENSION_ID}'] = () => require('../extensions/scratch3_${EXTENSION_ID}');${LF}${LF}class ExtensionManager {|g" node_modules/scratch-vm/src/extension-support/extension-manager.js_orig > node_modules/scratch-vm/src/extension-support/extension-manager.js

mkdir -p src/lib/libraries/extensions/${EXTENSION_ID}
cp ${EXTENSION_ID}/scratch-gui/src/lib/libraries/extensions/${EXTENSION_ID}/${EXTENSION_ID}_entry.png src/lib/libraries/extensions/${EXTENSION_ID}/
cp ${EXTENSION_ID}/scratch-gui/src/lib/libraries/extensions/${EXTENSION_ID}/${EXTENSION_ID}_inset.png src/lib/libraries/extensions/${EXTENSION_ID}/
mv src/lib/libraries/extensions/index.jsx src/lib/libraries/extensions/index.jsx_orig
DESCRIPTION="\
    {${LF}\
        name: (${LF}\
            <FormattedMessage${LF}\
                defaultMessage='${EXTENSION_NAME}'${LF}\
                description='Name of the extension'${LF}\
                id='gui.extension.${EXTENSION_ID}blocks.name'${LF}\
            />${LF}\
        ),${LF}\
        extensionId: '${EXTENSION_ID}',${LF}\
        collaborator: '${COLLABORATOR}',${LF}\
        iconURL: ${EXTENSION_ID}IconURL,${LF}\
        insetIconURL: ${EXTENSION_ID}InsetIconURL,${LF}\
        description: (${LF}\
            <FormattedMessage${LF}\
                defaultMessage='${EXTENSION_DESCRIPTION}'${LF}\
                description='Description of the extension'${LF}\
                id='gui.extension.${EXTENSION_ID}blocks.description'${LF}\
            />${LF}\
        ),${LF}\
        internetConnectionRequired: true,${LF}\
        featured: true,${LF}\
        translationMap: {${LF}\
          'es': {${LF}\
              'gui.extension.${EXTENSION_ID}blocks.name': 'Microbit Serial',${LF}\
              'gui.extension.${EXTENSION_ID}blocks.description': 'Extensión de MoonMakers para conectar la micro:bit a la computadora.'${LF}\
          },${LF}\
          'en': {${LF}\
              'gui.extension.${EXTENSION_ID}blocks.name': 'Microbit Serial',${LF}\
              'gui.extension.${EXTENSION_ID}blocks.description': 'MoonMakers extension to connect micro:bit to the computer.'${LF}\
          }${LF}\
        }${LF}\
    },"
sed -e "s|^export default \[$|import ${EXTENSION_ID}IconURL from './${EXTENSION_ID}/${EXTENSION_ID}_entry.png';${LF}import ${EXTENSION_ID}InsetIconURL from './${EXTENSION_ID}/${EXTENSION_ID}_inset.png';${LF}${LF}export default [${LF}${DESCRIPTION}|g" src/lib/libraries/extensions/index.jsx_orig > src/lib/libraries/extensions/index.jsx

mv webpack.config.js webpack.config.js_orig
CRYPTO_BROWSERIFY_FALLBACK="\
fallback: {${LF}\
                crypto: require.resolve('crypto-browserify'),\
"
sed -e "s|fallback: {|${CRYPTO_BROWSERIFY_FALLBACK}|g" webpack.config.js_orig > webpack.config.js
