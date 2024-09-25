$(document).ready(function() {
    // Función para manejar el clic en el encabezado del grupo para abrir o cerrar
    $('.group-header').click(function() {
        $(this).next('.group-items').slideToggle();
        $(this).find('.toggle-icon').text(function(_, text) {
            return text === '+' ? '-' : '+'; // Cambia el icono de "+" a "-" o viceversa
        });
    });
    
    // Array para almacenar las variables seleccionadas
    var selectedVariables = [];

    // Función para manejar la selección de variables
    $('#variables-list').on('click', 'li', function() {
        var variable = $(this).data('variable');

        // Verificar si la variable ya está seleccionada
        if (!selectedVariables.includes(variable)) {
            selectedVariables.push(variable);
            $(this).addClass('selected');
        } else {
            selectedVariables = selectedVariables.filter(function(item) {
                return item !== variable;
            });
            $(this).removeClass('selected');
        }
    });

    // Función para generar la gráfica
    $('#generate-chart-btn').click(function() {
        // Aquí puedes hacer la llamada al servidor con las variables seleccionadas
        // y luego generar la gráfica con los datos recibidos
        console.log('Variables seleccionadas:', selectedVariables);
        // Llama a tu función de Python para generar la gráfica con las variables seleccionadas

        // Realizar solicitud POST al servidor con las variables seleccionadas
        $.ajax({
            url: 'http://localhost:5000/generarGrafica', // Reemplaza 'url_del_servidor' con la URL de tu servicio en el servidor
            type: 'POST',
            data: JSON.stringify({ variables: selectedVariables }), // Envía las variables seleccionadas como datos de la solicitud
            contentType: 'application/json',
            success: function(response) {
                // Aquí puedes manejar la respuesta del servidor
                // Supongamos que el servidor devuelve la URL de la imagen generada
                var imageUrl = response.imageUrl;

                // Crea un enlace para descargar la imagen
                //var downloadLink = $('<a>').attr('href', imageUrl).text('Descargar Gráfica');
                //$('#download-link-container').empty().append(downloadLink);
                document.getElementById('imagen-grafica-container').innerHTML = '<img src="' + imageUrl + '" alt="Grafica' + selectedVariables + '">';
            


            },
            error: function(xhr, status, error) {
                console.error('Error al generar la gráfica:', error);
            }
        });
    });
});