$(window).on('load', function() {
    setInterval(function() {
        autosize($('textarea.autosize-textarea'));
        $('.random-input-button').click(function(event) {
            setTimeout(function() {
                autosize.update($('textarea.autosize-textarea'));
            }, 100);
        });
    }, 500);
})